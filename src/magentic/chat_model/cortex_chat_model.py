import asyncio
import json
import logging
import os
from collections.abc import Callable, Coroutine, Iterable, Iterator
from functools import singledispatch
from typing import Any, Generic, Literal, cast

import httpx
from pydantic import BaseModel
from typing_extensions import TypeVar

from magentic.chat_model.base import ChatModel, OutputT, aparse_stream, parse_stream
from magentic.chat_model.function_schema import (
    BaseFunctionSchema,
    FunctionCallFunctionSchema,
    get_async_function_schemas,
    get_function_schemas,
)
from magentic.chat_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
    _RawMessage,
)
from magentic.chat_model.stream import (
    AsyncOutputStream,
    FunctionCallChunk,
    OutputStream,
    StreamParser,
    StreamState,
)
from magentic.function_call import FunctionCall
from magentic.streaming import async_iter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolSpec(BaseModel):
    type: str = "generic"
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolUse(BaseModel):
    tool_use_id: str
    name: str
    input: dict[str, Any]


class ToolResults(BaseModel):
    tool_use_id: str
    name: str
    content: list[dict[str, Any]]  # list of {"type": "text", "text": "..."}


class ContentListItem(BaseModel):
    type: str  # "tool_use" or "tool_results"
    tool_use: ToolUse | None = None
    tool_results: ToolResults | None = None


class CortexMessage(BaseModel):
    role: Literal["user", "assistant", "function", "system"]
    content: str | None = None
    content_list: list[ContentListItem] | None = None


class Delta(BaseModel):
    content: str | None = None
    content_list: list[ContentListItem] | None = None


class Choice(BaseModel):
    delta: Delta


class StreamResponse(BaseModel):
    choices: list[Choice]


class CortexCompletionChunk(BaseModel):
    choices: list[dict[str, Any]]

    @property
    def delta(self) -> dict[str, Any]:
        return self.choices[0].get("delta", {}) if self.choices else {}


@singledispatch
def message_to_cortex_message(message: Message[Any]) -> dict[str, Any]:
    """Convert a Message to a Cortex-compatible message."""
    raise NotImplementedError(type(message))


@message_to_cortex_message.register(_RawMessage)
def _(message: _RawMessage[Any]) -> dict[str, Any]:
    return cast(dict[str, Any], message.content)


@message_to_cortex_message.register(UserMessage)
def _(message: UserMessage[Any]) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "user", "content": message.content}
    # Handle complex content if needed
    return {"role": "user", "content": str(message.content)}


def _function_call_to_tool_use(function_call: FunctionCall[Any]) -> ToolUse:
    function_schema = FunctionCallFunctionSchema(function_call.function)
    return ToolUse(
        tool_use_id=function_call._unique_id,
        name=function_schema.name,
        input=json.loads(function_schema.serialize_args(function_call)),
    )


@message_to_cortex_message.register(AssistantMessage)
def _(message: AssistantMessage[Any]) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": "assistant", "content": message.content}
    if isinstance(message.content, FunctionCall):
        tool_use = _function_call_to_tool_use(message.content)
        return {
            "role": "assistant",
            "content_list": [
                {
                    "type": "tool_use",
                    "tool_use": tool_use.model_dump(),
                }
            ],
        }
    return {"role": "assistant", "content": str(message.content)}


@message_to_cortex_message.register(SystemMessage)
def _(message: SystemMessage) -> dict[str, Any]:
    return {"role": "system", "content": message.content}


@message_to_cortex_message.register(ToolResultMessage)
def _(message: ToolResultMessage[Any]) -> dict[str, Any]:
    return {
        "role": "user",
        "content_list": [
            {
                "type": "tool_results",
                "tool_results": {
                    "tool_use_id": message.tool_call_id,
                    "name": getattr(message, "name", message.tool_call_id),
                    "content": [{"type": "text", "text": str(message.content)}],
                },
            }
        ],
    }


T = TypeVar("T")
BaseFunctionSchemaT = TypeVar("BaseFunctionSchemaT", bound=BaseFunctionSchema[Any])


class BaseFunctionToolSchema(Generic[BaseFunctionSchemaT]):
    def __init__(self, function_schema: BaseFunctionSchemaT):
        self._function_schema = function_schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_spec": {
                "type": "generic",
                "name": self._function_schema.name,
                "description": self._function_schema.description or "",
                "input_schema": self._function_schema.parameters,
            }
        }


class CortexStreamParser(StreamParser[CortexCompletionChunk]):
    def __init__(self) -> None:
        self._current_tool_call_id: str | None = None
        self._current_tool_call_name: str | None = None

    def is_content(self, item: CortexCompletionChunk) -> bool:
        return bool(item.delta.get("content"))

    def get_content(self, item: CortexCompletionChunk) -> str | None:
        content = item.delta.get("content")
        return content if content else None

    def is_tool_call(self, item: CortexCompletionChunk) -> bool:
        # Check if this is a tool_use delta
        delta_type = item.delta.get("type")
        return delta_type == "tool_use"

    def iter_tool_calls(
        self, item: CortexCompletionChunk
    ) -> Iterable[FunctionCallChunk]:
        if item.delta.get("type") == "tool_use":
            tool_use_id = item.delta.get("tool_use_id")
            name = item.delta.get("name")
            input_data = item.delta.get("input")

            if tool_use_id and name:
                # This is the start of a new tool call
                self._current_tool_call_id = tool_use_id
                self._current_tool_call_name = name
                chunk = FunctionCallChunk(
                    id=tool_use_id,
                    name=name,
                    args=None,
                )
                yield chunk
            elif (
                input_data
                and self._current_tool_call_id
                and self._current_tool_call_name
            ):
                # This is a continuation of the current tool call
                chunk = FunctionCallChunk(
                    id=self._current_tool_call_id,
                    name=self._current_tool_call_name,
                    args=(
                        input_data
                        if isinstance(input_data, str)
                        else json.dumps(input_data)
                    ),
                )
                yield chunk
            elif input_data:
                # Fallback: input data without context (shouldn't happen in well-formed responses)
                chunk = FunctionCallChunk(
                    id=None,
                    name=None,
                    args=(
                        input_data
                        if isinstance(input_data, str)
                        else json.dumps(input_data)
                    ),
                )
                yield chunk


class CortexStreamState(StreamState[CortexCompletionChunk]):
    def __init__(self) -> None:
        self.usage_ref: list[Usage] = []

    def update(self, item: CortexCompletionChunk) -> None:
        # Leaving it empty because cortex doesn't provide usage in streaming
        pass

    @property
    def current_message_snapshot(self) -> Message[Any]:
        """Return a basic message snapshot."""
        return _RawMessage({"role": "assistant", "content": ""})


def _parse_sse_response(response_text: str) -> Iterator[CortexCompletionChunk]:
    """Parse SSE response into CortexCompletionChunk objects."""
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            json_data = line[6:]  # Remove 'data: ' prefix
            if json_data.strip():  # Skip empty data lines
                try:
                    chunk_data = json.loads(json_data)
                    yield CortexCompletionChunk(**chunk_data)
                except json.JSONDecodeError:
                    continue  # Skip malformed chunks


class SnowflakeChatModel(ChatModel):
    """
    ChatModel implementation for Snowflake Cortex, compatible with Magentic decorators.
    Supports streaming, function calls, and structured outputs.
    """

    def __init__(
        self,
        model: str,
        account: str | None = None,
        token: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        *,
        guardrails_enabled: bool = False,
        response_when_unsafe: bool = True,
    ):
        self.model = model
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.token = token or os.getenv("SNOWFLAKE_PAT")
        if not self.account:
            _no_account_msg = (
                "Account must be provided or set in SNOWFLAKE_ACCOUNT env var"
            )
            raise ValueError(_no_account_msg)
        if not self.token:
            _no_token_msg = "Programmatic Access Token must be provided or set in SNOWFLAKE_PAT env var"  # noqa: S105
            raise ValueError(_no_token_msg)
        self.base_url = f"https://{self.account}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        self.timeout = httpx.Timeout(30.0)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.guardrails = {
            "enabled": guardrails_enabled,
            "response_when_unsafe": response_when_unsafe,
        }

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            return response

    def _run_async_in_sync_context(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Run an async coroutine in either sync or async context."""
        try:
            # Check if we're already in an async context
            asyncio.get_running_loop()
            # If so, we need to run the coroutine synchronously
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(coro)

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        functions: list[Callable[..., Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if functions:
            tool_schemas = [
                BaseFunctionToolSchema(schema)
                for schema in get_function_schemas(functions, [])
            ]
            payload["tools"] = [schema.to_dict() for schema in tool_schemas]
        return payload

    def complete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Request an LLM message."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[OutputT]], [str])

        # Convert messages
        converted_messages = [message_to_cortex_message(m) for m in messages]

        functions_list = list(functions) if functions else None
        payload = self._build_payload(converted_messages, functions_list)

        # Run async post in sync context (handles both sync and async callers)
        response = self._run_async_in_sync_context(self._post(payload))

        # Parse SSE response into chunks
        response_text = response.text
        chunks = list(_parse_sse_response(response_text))

        # Check if we expect function calls and the first chunk is text
        # If so, look ahead to see if there are tool calls in the stream
        has_function_call_type = any(
            "FunctionCall" in str(type_) for type_ in output_types
        )

        if (
            has_function_call_type
            and chunks
            and chunks[0].delta.get("type") == "text"
            and any(chunk.delta.get("type") == "tool_use" for chunk in chunks)
        ):
            # Find the first tool call chunk
            tool_call_chunk = next(
                (chunk for chunk in chunks if chunk.delta.get("type") == "tool_use"),
                None,
            )
            if tool_call_chunk:
                # Process the tool call directly
                function_schemas = get_function_schemas(functions_list, output_types)
                parser = CortexStreamParser()
                state = CortexStreamState()

                # Create a stream starting from the tool call chunk
                tool_call_index = chunks.index(tool_call_chunk)
                tool_call_stream = iter(chunks[tool_call_index:])

                stream = OutputStream(
                    tool_call_stream,
                    function_schemas=function_schemas,
                    parser=parser,
                    state=state,
                )

                return AssistantMessage._with_usage(
                    parse_stream(stream, output_types), usage_ref=stream.usage_ref
                )

        # Use existing streaming infrastructure for normal cases
        function_schemas = get_function_schemas(functions_list, output_types)
        stream = OutputStream(
            iter(chunks),
            function_schemas=function_schemas,
            parser=CortexStreamParser(),
            state=CortexStreamState(),
        )

        return AssistantMessage._with_usage(
            parse_stream(stream, output_types), usage_ref=stream.usage_ref
        )

    async def acomplete(
        self,
        messages: Iterable[Message[Any]],
        functions: Iterable[Callable[..., Any]] | None = None,
        output_types: Iterable[type[OutputT]] | None = None,
        *,
        stop: list[str] | None = None,
    ) -> AssistantMessage[OutputT]:
        """Async version of `complete`."""
        if output_types is None:
            output_types = [] if functions else cast(list[type[OutputT]], [str])

        # Convert messages
        converted_messages = [message_to_cortex_message(m) for m in messages]

        functions_list = list(functions) if functions else None
        payload = self._build_payload(converted_messages, functions_list)

        response = await self._post(payload)

        # Parse SSE response into chunks
        response_text = response.text
        chunks = list(_parse_sse_response(response_text))

        # Check if we expect function calls and the first chunk is text
        # If so, look ahead to see if there are tool calls in the stream
        has_function_call_type = any(
            "FunctionCall" in str(type_) for type_ in output_types
        )

        if (
            has_function_call_type
            and chunks
            and chunks[0].delta.get("type") == "text"
            and any(chunk.delta.get("type") == "tool_use" for chunk in chunks)
        ):
            # Find the first tool call chunk
            tool_call_chunk = next(
                (chunk for chunk in chunks if chunk.delta.get("type") == "tool_use"),
                None,
            )
            if tool_call_chunk:
                # Process the tool call directly
                function_schemas = get_async_function_schemas(
                    functions_list, output_types
                )
                parser = CortexStreamParser()
                state = CortexStreamState()

                # Create a stream starting from the tool call chunk
                tool_call_index = chunks.index(tool_call_chunk)
                tool_call_stream = async_iter(chunks[tool_call_index:])

                stream = AsyncOutputStream(
                    tool_call_stream,
                    function_schemas=function_schemas,
                    parser=parser,
                    state=state,
                )

                return AssistantMessage._with_usage(
                    await aparse_stream(stream, output_types),
                    usage_ref=stream.usage_ref,
                )

        # Use existing streaming infrastructure for normal cases
        function_schemas = get_async_function_schemas(functions_list, output_types)
        stream = AsyncOutputStream(
            async_iter(chunks),
            function_schemas=function_schemas,
            parser=CortexStreamParser(),
            state=CortexStreamState(),
        )

        return AssistantMessage._with_usage(
            await aparse_stream(stream, output_types), usage_ref=stream.usage_ref
        )
