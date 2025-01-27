import inspect
import json
import logging
from abc import ABC
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field

from ironhide.settings import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)s  %(filename)s  %(funcName)s  %(message)s",
)

COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


class PropertyDefinition(BaseModel):
    type: str
    description: str


class ParametersDefinition(BaseModel):
    type: str = "object"
    properties: dict[str, PropertyDefinition]
    required: list[str]
    additionalProperties: bool = False


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: ParametersDefinition
    strict: bool = True


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class Headers(BaseModel):
    content_type: str = Field(
        alias="Content-Type",
        default="application/json",
    )
    authorization: str = Field(
        alias="Authorization",
        default=f"Bearer {settings.openai_api_key}",
    )


class Role(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"


class PromptTokensDetails(BaseModel):
    cached_tokens: int
    audio_tokens: int


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: ToolFunction


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    model_config = {"use_enum_values": True}


class Choice(BaseModel):
    index: int
    message: Message
    logprobs: dict[str, Any] | None = None
    finish_reason: str


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    service_tier: str
    system_fingerprint: str


class JsonSchema(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class ResponseFormat(BaseModel):
    type: str = "json_schema"
    json_schema: JsonSchema


class Data(BaseModel):
    model: str
    messages: list[Message]
    response_format: ResponseFormat | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["none", "auto", "required"] | None = None


class Approval(BaseModel):
    is_approved: bool


class BaseAgent(ABC):
    model: str
    instructions: str | None = None
    response_format: type[BaseModel] | None = None
    chain_of_thought: tuple[str, ...] | None = None
    feedback_loop: str | None = None

    def __init__(
        self,
        instructions: str | None = None,
        response_format: type[BaseModel] | None = None,
        chain_of_thought: tuple[str, ...] | None = None,
        feedback_loop: str | None = None,
        model: str | None = None,
    ) -> None:
        self.instructions = instructions or getattr(self, "instructions", None)
        self.response_format = response_format or getattr(self, "response_format", None)
        self.chain_of_thought = chain_of_thought or getattr(
            self,
            "chain_of_thought",
            None,
        )
        self.feedback_loop = feedback_loop or getattr(self, "feedback_loop", None)
        self.model = model or getattr(self, "model", None) or settings.default_model
        self.messages: list[Message] = []
        self.dict_tool: dict[str, Any] = {}
        self.tools = self._generate_tools()
        if self.instructions:
            self.add_message(Message(role=Role.system, content=self.instructions))
        self.client = httpx.AsyncClient()
        self.headers = Headers()

    def _add_additional_properties(self, obj: dict) -> None:
        obj["additionalProperties"] = False
        if "$defs" in obj:
            for def_schema in obj["$defs"].values():
                if isinstance(def_schema, dict):
                    self._add_additional_properties(def_schema)

    def _make_response_format_section(
        self,
        response_format: type[BaseModel] | None,
    ) -> ResponseFormat | None:
        if response_format is None:
            return None
        schema = response_format.model_json_schema()
        self._add_additional_properties(schema)
        return ResponseFormat(
            json_schema=JsonSchema(
                name=schema["title"],
                schema=schema,
            ),
        )

    def _generate_tools(self) -> list[ToolDefinition]:
        tools = []
        json_type_mapping = {
            str: "string",
            int: "number",
            float: "number",
            bool: "boolean",
        }
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if not getattr(method, "is_tool", False):
                continue
            properties = {
                param_name: PropertyDefinition(
                    type=json_type_mapping.get(param.annotation, "string"),
                    description=(
                        param.annotation.__metadata__[0]
                        if getattr(param.annotation, "__metadata__", None)
                        else ""
                    ),
                )
                for param_name, param in inspect.signature(method).parameters.items()
                if param_name != "self"
            }
            required = [
                param_name
                for param_name, param in inspect.signature(method).parameters.items()
                if param.default is param.empty and param_name != "self"
            ]
            tools.append(
                ToolDefinition(
                    function=FunctionDefinition(
                        name=name,
                        description=(inspect.getdoc(method) or "").strip(),
                        parameters=ParametersDefinition(
                            properties=properties, required=required
                        ),
                    ),
                ),
            )
            self.dict_tool[name] = method
        return tools

    async def _call_function(self, name: str, args: dict[str, Any]) -> Any:
        selected_tool = self.dict_tool[name]
        if inspect.iscoroutinefunction(selected_tool):
            return await selected_tool(**args)
        return selected_tool(**args)

    async def _api_call(
        self,
        *,
        is_thought: bool = False,
        is_approval: bool = False,
        response_format: type[BaseModel] | None = None,
    ) -> Message:
        current_response_format = (
            None
            if is_thought
            else Approval
            if is_approval
            else response_format or self.response_format
        )
        data = Data(
            model=self.model,
            messages=self.messages,
            response_format=self._make_response_format_section(current_response_format)
            if not is_thought
            else None,
            tools=self.tools or None,
            tool_choice=None
            if not self.tools
            else "none"
            if is_thought or is_approval
            else "auto",
        )
        response = await self.client.post(
            COMPLETIONS_URL,
            headers=self.headers.model_dump(by_alias=True),
            json=data.model_dump(by_alias=True),
            timeout=settings.timeout,
        )
        if response.status_code != 200:
            logging.error(json.dumps(data.model_dump(by_alias=True), indent=4))
            logging.error(response.text)
            raise Exception(response.text)
        completion = ChatCompletion(**response.json())
        message = completion.choices[0].message
        self.add_message(message)
        return message

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        logging.info(json.dumps(message.model_dump(exclude_none=True), indent=4))

    async def chat(
        self,
        input_message: str,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        self.add_message(Message(role=Role.user, content=input_message))
        is_approved = False
        while not is_approved:
            # Chain of thought
            if self.chain_of_thought:
                for thought in self.chain_of_thought:
                    self.add_message(Message(role=Role.system, content=thought))
                    await self._api_call(is_thought=True)

            # Tool calls
            message = await self._api_call(response_format=response_format)
            tool_calls = message.tool_calls
            while tool_calls:
                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    id_ = tool_call.id
                    result = await self._call_function(name, args)
                    self.add_message(
                        Message(role=Role.tool, content=str(result), tool_call_id=id_),
                    )
                message = await self._api_call(response_format=response_format)
                tool_calls = message.tool_calls

            # Feedback loop
            if self.feedback_loop:
                self.add_message(Message(role=Role.system, content=self.feedback_loop))
                await self._api_call(is_thought=True)
                message = await self._api_call(is_approval=True)
                is_approved = Approval(**json.loads(message.content or "")).is_approved
            else:
                is_approved = True

        # Response Message
        message = await self._api_call(response_format=response_format)
        return message.content or ""


F = TypeVar("F", bound=Callable[..., Any])


def tool(func: F) -> F:
    func.is_tool = True  # type: ignore[attr-defined]
    return func
