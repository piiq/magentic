import os

import pytest

from magentic.chat_model.cortex_chat_model import (
    SnowflakeChatModel,
    message_to_cortex_message,
)
from magentic.chat_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from magentic.function_call import FunctionCall


def plus(a: int, b: int) -> int:
    return a + b


message_to_cortex_message_test_cases = [
    (UserMessage("Hello"), {"role": "user", "content": "Hello"}),
    (AssistantMessage("Hello"), {"role": "assistant", "content": "Hello"}),
    (
        AssistantMessage(FunctionCall(plus, 1, 2)),
        {
            "role": "assistant",
            "content_list": [
                {
                    "type": "tool_use",
                    "tool_use": {
                        "tool_use_id": "test_id",
                        "name": "plus",
                        "input": {"a": 1, "b": 2},
                    },
                }
            ],
        },
    ),
    (
        SystemMessage("You are a helpful assistant."),
        {"role": "system", "content": "You are a helpful assistant."},
    ),
    (
        ToolResultMessage("Result", "tool_call_id"),
        {
            "role": "user",
            "content_list": [
                {
                    "type": "tool_results",
                    "tool_results": {
                        "tool_use_id": "tool_call_id",
                        "name": "tool_call_id",
                        "content": [{"type": "text", "text": "Result"}],
                    },
                }
            ],
        },
    ),
]


@pytest.mark.parametrize(
    ("message", "expected_cortex_message"), message_to_cortex_message_test_cases
)
def test_message_to_cortex_message(message, expected_cortex_message):
    result = message_to_cortex_message(message)
    # For function calls, the ID is generated, so we check structure
    if isinstance(message.content, FunctionCall):
        assert result["role"] == expected_cortex_message["role"]
        assert "content_list" in result
        assert result["content_list"][0]["type"] == "tool_use"
        assert result["content_list"][0]["tool_use"]["name"] == "plus"
        assert result["content_list"][0]["tool_use"]["input"] == {"a": 1, "b": 2}
    else:
        assert result == expected_cortex_message


@pytest.mark.cortex
def test_cortex_chat_model_complete():
    chat_model = SnowflakeChatModel(
        model="claude-3-5-sonnet",
        account=os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        token=os.getenv("SNOWFLAKE_PAT"),
    )
    message = chat_model.complete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.cortex
def test_cortex_chat_model_complete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = SnowflakeChatModel(
        model="claude-3-5-sonnet",
        account=os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        token=os.getenv("SNOWFLAKE_PAT"),
    )
    message = chat_model.complete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)


@pytest.mark.cortex
async def test_cortex_chat_model_acomplete():
    chat_model = SnowflakeChatModel(
        model="claude-3-5-sonnet",
        account=os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        token=os.getenv("SNOWFLAKE_PAT"),
    )
    message = await chat_model.acomplete(messages=[UserMessage("Say hello!")])
    assert isinstance(message.content, str)


@pytest.mark.cortex
async def test_cortex_chat_model_acomplete_function_call():
    def plus(a: int, b: int) -> int:
        """Sum two numbers."""
        return a + b

    chat_model = SnowflakeChatModel(
        model="claude-3-5-sonnet",
        account=os.getenv("SNOWFLAKE_ACCOUNT", "test_account"),
        token=os.getenv("SNOWFLAKE_PAT"),
    )
    message = await chat_model.acomplete(
        messages=[UserMessage("Use the tool to sum 1 and 2")],
        functions=[plus],
        output_types=[FunctionCall[int]],
    )
    assert isinstance(message.content, FunctionCall)
