import base64
import inspect
import json
import logging
from abc import ABC
from asyncio import sleep
from collections.abc import Buffer, Callable
from enum import Enum
from typing import Any, Literal, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field

from ironhide.settings import settings

logger = logging.getLogger(__name__)

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


class TextContent(BaseModel):
    type: str = "text"
    text: str


class ImageUrlContent(BaseModel):
    type: str = "image_url"
    image_url: dict[str, str]


class Message(BaseModel):
    role: Role
    content: str | list[TextContent | ImageUrlContent] | None = None
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


class Error(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None


class ErrorResponse(BaseModel):
    error: Error


class Reason(BaseModel):
    thought: str


class Approval(BaseModel):
    is_approved: bool


async def audio_transcription(files: RequestFiles) -> str:
    transcription_url = "https://api.openai.com/v1/audio/transcriptions"
    transcription_headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    async with httpx.AsyncClient() as client:
        data = {"model": settings.openai_audio_to_text_model_id}
        transcription_response = await client.post(
            transcription_url,
            headers=transcription_headers,
            files=files,
            data=data,
        )
    return str(transcription_response.json().get("text", ""))


T = TypeVar("T", bound=BaseModel)


class BaseAgent(ABC):
    model: str
    instructions: str | None = None
    chain_of_thought: tuple[str, ...] | None = None
    feedback_loop: str | None = None
    messages: list[Message] = []

    def __init__(
        self,
        instructions: str | None = None,
        chain_of_thought: tuple[str, ...] | None = None,
        feedback_loop: str | None = None,
        model: str | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        self.instructions = instructions or getattr(self, "instructions", None)
        self.chain_of_thought = chain_of_thought or getattr(
            self,
            "chain_of_thought",
            None,
        )
        self.feedback_loop = feedback_loop or getattr(self, "feedback_loop", None)
        self.model = model or getattr(self, "model", None) or settings.default_model
        self.messages = (
            messages or getattr(self, "messages", None) or self._get_history()
        )
        self.dict_tool: dict[str, Any] = {}
        self.tools = self._generate_tools()
        if self.instructions:
            self.add_message(
                Message(role=Role.system, content=self.instructions),
                in_begin=True,
            )
        self.client = httpx.AsyncClient()
        self.headers = Headers()

    def _get_history(self) -> list[Message]:
        return []

    def _remove_defaults(self, schema: dict[str, Any]) -> None:
        if isinstance(schema, dict):
            schema.pop("default", None)
            schema.pop("format", None)
            for value in schema.values():
                if isinstance(value, dict):
                    self._remove_defaults(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._remove_defaults(item)

    def _add_additional_properties(self, obj: dict[str, Any]) -> None:
        obj["additionalProperties"] = False
        if "properties" in obj:
            for def_schema in obj["properties"].values():
                if isinstance(def_schema, dict):
                    self._add_additional_properties(def_schema)

    def _remove_defs(self, schema: dict[str, Any]) -> None:
        if schema.get("$defs"):
            for i in schema["properties"]:
                property_ref = schema["properties"][i]["$ref"]
                for j in schema["$defs"]:
                    clean_property = property_ref.replace("#/$defs/", "")
                    if clean_property == j:
                        schema["properties"][i] = schema["$defs"][j]
            schema.pop("$defs")

    def _make_response_format_section(
        self,
        response_format: type[BaseModel] | None,
    ) -> ResponseFormat | None:
        if response_format is None:
            return None
        schema = response_format.model_json_schema()
        self._remove_defs(schema)
        self._remove_defaults(schema)
        self._add_additional_properties(schema)
        properties = schema.get("properties", {})
        if "required" not in schema:
            schema["required"] = list(properties.keys())

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
                            properties=properties,
                            required=required,
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
            Reason if is_thought else Approval if is_approval else response_format
        )
        data = Data(
            model=self.model,
            messages=self.messages,
            response_format=self._make_response_format_section(current_response_format),
            tools=self.tools or None,
            tool_choice=None
            if not self.tools
            else "none"
            if is_thought or is_approval
            else "auto",
        )
        logger.debug(
            data.model_dump_json(by_alias=True, exclude_none=True, indent=4),
        )
        while True:
            response = await self.client.post(
                COMPLETIONS_URL,
                headers=self.headers.model_dump(by_alias=True),
                json=data.model_dump(
                    by_alias=True,
                    mode="json",
                    exclude_none=True,
                ),
                timeout=settings.timeout,
            )
            if response.status_code != 200:
                logger.error(
                    data.model_dump_json(by_alias=True, exclude_none=True, indent=4),
                )
                error_response = ErrorResponse.model_validate_json(response.text)
                logger.error(error_response.error.message)
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded. Retrying in 10 seconds...")
                    await sleep(10)
                    continue
                else:
                    raise Exception(error_response.error.message)
            break
        completion = ChatCompletion(**response.json())
        message = completion.choices[0].message
        self.add_message(message)
        return message

    def add_message(self, message: Message, *, in_begin: bool = False) -> None:
        if in_begin:
            self.messages.insert(0, message)
        else:
            self.messages.append(message)
        logger.info(message.model_dump_json(by_alias=True, exclude_none=True, indent=4))

    async def _context_provider(self, input_message: str) -> str:
        return input_message

    async def _base_chat(
        self,
        input_message: str | RequestFiles,
        response_format: type[T] | None = None,
        files: RequestFiles | None = None,
    ) -> str:
        # Handle audio transcription if input_message is not a string
        processed_message: str = (
            await audio_transcription(input_message)
            if not isinstance(input_message, str)
            else input_message
        )

        # Handle image transcription if files are provided
        if files:
            (name, file_bytes, mime) = files["file"]  # type: ignore
            if isinstance(file_bytes, Buffer):
                base64_image = base64.b64encode(file_bytes).decode("utf-8")
                content_items: list[TextContent | ImageUrlContent] = [
                    TextContent(text=processed_message),
                    ImageUrlContent(
                        image_url={"url": f"data:{mime!s};base64,{base64_image}"},
                    ),
                ]
                self.add_message(
                    Message(role=Role.user, content=content_items),
                )
        else:
            self.add_message(
                Message(
                    role=Role.user,
                    content=await self._context_provider(processed_message),
                ),
            )

        is_approved = False
        message: Message | None = None
        while not is_approved:
            # Chain of thought
            if self.chain_of_thought:
                for thought in self.chain_of_thought:
                    self.add_message(Message(role=Role.system, content=thought))
                    await self._api_call(is_thought=True)
            message = await self._api_call(response_format=response_format)

            # Tool calls
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
                is_approved = Approval.model_validate_json(
                    str(message.content) or "",
                ).is_approved
                if is_approved:
                    message = await self._api_call(response_format=response_format)
                    break
            else:
                break
        content = ""
        if message:
            content = str(message.content)
        return content

    async def chat(
        self,
        input_message: str | RequestFiles,
        files: RequestFiles | None = None,
    ) -> str:
        return await self._base_chat(input_message=input_message, files=files)

    async def structred_chat(
        self,
        input_message: str | RequestFiles,
        response_format: type[T],
        files: RequestFiles | None = None,
    ) -> T:
        content = await self._base_chat(
            input_message=input_message, files=files, response_format=response_format
        )
        return response_format.model_validate_json(content)


F = TypeVar("F", bound=Callable[..., Any])


def tool(func: F) -> F:
    func.is_tool = True  # type: ignore[attr-defined]
    return func
