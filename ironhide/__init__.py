"""Ironhide: A framework for building AI agents with chat capabilities.

This package provides tools and abstractions for creating AI agents that can engage in chat-based
interactions, handle structured responses, and utilize various tools and feedback mechanisms.
"""

import base64
import inspect
import json
import logging
from abc import ABC
from asyncio import sleep
from collections.abc import Buffer, Callable
from enum import Enum
from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field

from ironhide.models import Provider
from ironhide.settings import settings

logger = logging.getLogger(__name__)


class _PropertyDefinition(BaseModel):
    type: str
    description: str


class _ParametersDefinition(BaseModel):
    type: str = "object"
    properties: dict[str, _PropertyDefinition]
    required: list[str]
    additional_properties: bool = Field(alias="additionalProperties", default=False)


class _FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: _ParametersDefinition
    strict: bool = True


class _ToolDefinition(BaseModel):
    type: str = "function"
    function: _FunctionDefinition


class _Headers(BaseModel):
    content_type: str = Field(
        alias="Content-Type",
        default="application/json",
    )
    authorization: str = Field(alias="Authorization")


class _Role(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    tool = "tool"


class _PromptTokensDetails(BaseModel):
    cached_tokens: int
    audio_tokens: int


class _CompletionTokensDetails(BaseModel):
    reasoning_tokens: int
    audio_tokens: int
    accepted_prediction_tokens: int
    rejected_prediction_tokens: int


class _Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: _PromptTokensDetails
    completion_tokens_details: _CompletionTokensDetails


class _ToolFunction(BaseModel):
    name: str
    arguments: str


class _ToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: _ToolFunction


class _TextContent(BaseModel):
    type: str = "text"
    text: str


class _ImageUrlContent(BaseModel):
    type: str = "image_url"
    image_url: dict[str, str]


class _Message(BaseModel):
    role: _Role
    content: str | list[_TextContent | _ImageUrlContent] | None = None
    tool_calls: list[_ToolCall] | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    model_config = {"use_enum_values": True}


class _Choice(BaseModel):
    index: int
    message: _Message
    logprobs: dict[str, Any] | None = None
    finish_reason: str


class _ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[_Choice]
    usage: _Usage
    service_tier: str
    system_fingerprint: str


class _JsonSchema(BaseModel):
    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class _ResponseFormat(BaseModel):
    type: str = "json_schema"
    json_schema: _JsonSchema


class _Data(BaseModel):
    model: str
    messages: list[_Message]
    response_format: _ResponseFormat | None = None
    tools: list[_ToolDefinition] | None = None
    tool_choice: Literal["none", "auto", "required"] | None = None


class _Error(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None


class _ErrorResponse(BaseModel):
    error: _Error


class _Reason(BaseModel):
    thought: str


class _Approval(BaseModel):
    is_approved: bool


async def audio_transcription(files: RequestFiles, api_key: str) -> str:
    """Transcribes audio files to text using the OpenAI API.

    Args:
        files: RequestFiles object containing the audio file to transcribe.
        api_key: OpenAI API key for authentication.

    Returns:
        The transcribed text as a string.

    """
    transcription_url = "https://api.openai.com/v1/audio/transcriptions"
    transcription_headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient() as client:
        data = {"model": settings.openai_audio_to_text_model_id}
        transcription_response = await client.post(
            transcription_url,
            headers=transcription_headers,
            files=files,
            data=data,
        )
    return str(transcription_response.json().get("text", ""))


def get_prompt(file_path: str, prompt: str) -> str:
    """Read and return the content of a prompt file.

    Args:
        file_path: Path to the main file.
        prompt: Name of the prompt file (relative to the main file's directory).

    Returns:
        The content of the prompt file.

    """
    system_file_path = Path(file_path).parent / prompt
    with system_file_path.open(encoding="utf-8") as system_file:
        return system_file.read()


T = TypeVar("T", bound=BaseModel)


class OpenaiAgent(ABC):
    """Base class for implementing AI agents with chat capabilities.

    This abstract class provides the foundation for creating AI agents that can engage in chat-based interactions,
    handle structured responses, and utilize various tools and feedback mechanisms.

    Class Variables:
        model (str): The AI model identifier to be used.
        provider (Provider): The service provider for the AI model.
        instructions (str | None): System instructions for the agent.
        chain_of_thought (tuple[str, ...] | None): Sequential prompts for thought process.
        feedback_loop (str | None): Prompt for feedback evaluation.
        messages (list[_Message]): History of chat messages.
        instructions (str | None): Initial system instructions for the agent.
        chain_of_thought (tuple[str, ...] | None): Sequence of thought process prompts.
        feedback_loop (str | None): Feedback evaluation prompt.
        model (str | None): AI model identifier.
        provider (Provider | None): Service provider for the AI model.
        messages (list[_Message] | None): Initial chat message history.

    Methods:
        chat(input_message: str | RequestFiles, files: RequestFiles | None = None) -> str:
            Handle a chat interaction with optional audio or image processing.
                input_message: User's input message (text or audio files)
                files: Optional image files for the chat
                str: The assistant's response
        structured_chat(input_message: str | RequestFiles, response_format: type[T], files: RequestFiles | None = None) -> T:
            Handle a chat interaction with structured response validation.
                input_message: User's input message (text or audio files)
                response_format: Pydantic model for response validation
                files: Optional image files for the chat
                T: The assistant's response as a validated Pydantic model instance.

    """

    model: str
    api_key: str
    provider: Provider
    instructions: str | None = None
    chain_of_thought: tuple[str, ...] | None = None
    feedback_loop: str | None = None
    messages: list[_Message]

    def __init__(
        self,
        instructions: str | None = None,
        chain_of_thought: tuple[str, ...] | None = None,
        feedback_loop: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        provider: Provider | None = None,
        messages: list[_Message] | None = None,
    ) -> None:
        """Initialize the BaseAgent with optional configuration parameters.

        Args:
            instructions: Initial system instructions for the agent.
            chain_of_thought: Sequence of thought process prompts.
            feedback_loop: Feedback evaluation prompt.
            model: AI model identifier.
            provider: Service provider for the AI model.
            messages: Initial chat message history.

        """
        self.instructions = instructions or getattr(self, "instructions", None)
        self.chain_of_thought = chain_of_thought or getattr(
            self,
            "chain_of_thought",
            None,
        )
        self.feedback_loop = feedback_loop or getattr(self, "feedback_loop", None)
        self.provider = (
            provider or getattr(self, "provider", None) or settings.default_provider
        )
        self.model = model or getattr(self, "model", None) or settings.default_model
        self.api_key = (
            api_key or getattr(self, "api_key", None) or settings.default_api_key
        )
        self.messages = (
            messages or getattr(self, "messages", None) or self._get_history()
        )
        self.dict_tool: dict[str, Any] = {}
        self.tools = self._generate_tools()
        if self.instructions:
            self._add_message(
                _Message(role=_Role.system, content=self.instructions),
                in_begin=True,
            )
        self.client = httpx.AsyncClient()
        self.headers = _Headers(Authorization=f"Bearer {self.api_key}")

    def _get_history(self) -> list[_Message]:
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
    ) -> _ResponseFormat | None:
        if response_format is None:
            return None
        schema = response_format.model_json_schema()
        self._remove_defs(schema)
        self._remove_defaults(schema)
        self._add_additional_properties(schema)
        properties = schema.get("properties", {})
        if "required" not in schema:
            schema["required"] = list(properties.keys())

        return _ResponseFormat(
            json_schema=_JsonSchema(
                name=schema["title"],
                schema=schema,
            ),
        )

    def _generate_tools(self) -> list[_ToolDefinition]:
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
                param_name: _PropertyDefinition(
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
                _ToolDefinition(
                    function=_FunctionDefinition(
                        name=name,
                        description=(inspect.getdoc(method) or "").strip(),
                        parameters=_ParametersDefinition(
                            properties=properties,
                            required=required,
                        ),
                    ),
                ),
            )
            self.dict_tool[name] = method
        return tools

    async def _call_function(self, name: str, args: dict[str, Any]) -> Any:  # noqa: ANN401
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
    ) -> _Message:
        current_response_format = (
            _Reason if is_thought else _Approval if is_approval else response_format
        )
        data = _Data(
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
                settings.openai_url + "/chat/completions",
                headers=self.headers.model_dump(by_alias=True),
                json=data.model_dump(
                    by_alias=True,
                    mode="json",
                    exclude_none=True,
                ),
                timeout=settings.request_timeout,
            )
            if response.status_code != HTTPStatus.OK.value:
                logger.error(
                    data.model_dump_json(by_alias=True, exclude_none=True, indent=4),
                )
                error_response = _ErrorResponse.model_validate_json(response.text)
                logger.error(error_response.error.message)
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS.value:
                    logger.warning("Rate limit exceeded. Retrying in 10 seconds...")
                    # TODO: esse valor deve ser capturado da string do erro
                    # TODO: verificar se a mensagem do erro é isso: You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.
                    await sleep(10)
                    continue
                raise Exception(error_response.error.message)
            break
        completion = _ChatCompletion(**response.json())
        message = completion.choices[0].message
        self._add_message(message)
        return message

    def _add_message(self, message: _Message, *, in_begin: bool = False) -> None:
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
        processed_message: str = (
            await audio_transcription(input_message, self.api_key)
            if not isinstance(input_message, str)
            else input_message
        )

        if files:
            await self._handle_image_message(processed_message, files)
        else:
            self._add_message(
                _Message(
                    role=_Role.user,
                    content=await self._context_provider(processed_message),
                ),
            )

        is_approved = False
        message: _Message | None = None
        while not is_approved:
            await self._handle_chain_of_thought()
            message = await self._api_call(response_format=response_format)
            message = await self._handle_tool_calls(message, response_format)
            if self.feedback_loop:
                is_approved, message = await self._handle_feedback_loop(
                    message, response_format
                )
                if is_approved:
                    break
            else:
                break
        content = ""
        if message:
            content = str(message.content)
        return content

    async def _handle_image_message(
        self, processed_message: str, files: RequestFiles
    ) -> None:
        name, file_bytes, mime = files["file"]  # type: ignore[misc, call-overload]
        if isinstance(file_bytes, Buffer):
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            content_items: list[_TextContent | _ImageUrlContent] = [
                _TextContent(text=processed_message),
                _ImageUrlContent(
                    image_url={"url": f"data:{mime!s};base64,{base64_image}"},
                ),
            ]
            self._add_message(
                _Message(role=_Role.user, content=content_items),
            )

    async def _handle_chain_of_thought(self) -> None:
        if self.chain_of_thought:
            for thought in self.chain_of_thought:
                self._add_message(_Message(role=_Role.system, content=thought))
                await self._api_call(is_thought=True)

    async def _handle_tool_calls(
        self, message: _Message, response_format: type[T] | None
    ) -> _Message:
        tool_calls = message.tool_calls
        while tool_calls:
            for tool_call in tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                id_ = tool_call.id
                result = await self._call_function(name, args)
                self._add_message(
                    _Message(
                        role=_Role.tool,
                        content=str(result),
                        tool_call_id=id_,
                    ),
                )
            message = await self._api_call(response_format=response_format)
            tool_calls = message.tool_calls
        return message

    async def _handle_feedback_loop(
        self, message: _Message, response_format: type[T] | None
    ) -> tuple[bool, _Message]:
        self._add_message(
            _Message(role=_Role.system, content=self.feedback_loop),
        )
        await self._api_call(is_thought=True)
        message = await self._api_call(is_approval=True)
        is_approved = _Approval.model_validate_json(
            str(message.content) or "",
        ).is_approved
        if is_approved:
            message = await self._api_call(response_format=response_format)
        return is_approved, message

    async def chat(
        self,
        input_message: str | RequestFiles,
        files: RequestFiles | None = None,
    ) -> str:
        """Handle a chat interaction, optionally processing audio or image files.

        Args:
            input_message: The user's input message, which can be text or audio files.
            files: Optional image files to be included in the chat.

        Returns:
            The assistant's response as a string.

        """
        return await self._base_chat(input_message=input_message, files=files)

    async def structured_chat(
        self,
        input_message: str | RequestFiles,
        response_format: type[T],
        files: RequestFiles | None = None,
    ) -> T:
        """Handle a chat interaction with a structured response.

        Args:
            input_message: The user's input message, which can be text or audio files.
            response_format: The Pydantic model to validate and parse the response.
            files: Optional image files to be included in the chat.

        Returns:
            The assistant's response as a Pydantic model instance.

        """
        content = await self._base_chat(
            input_message=input_message,
            files=files,
            response_format=response_format,
        )
        # TODO: Adicionar try catch na validação
        return response_format.model_validate_json(content)


F = TypeVar("F", bound=Callable[..., Any])


def tool(func: F) -> F:
    """Mark a method as a tool that can be called by the AI model."""
    func.is_tool = True  # type: ignore[attr-defined]
    return func
