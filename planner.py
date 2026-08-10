"""
Planning layer: decides whether the agent's next step is a tool call or a
direct answer, using OpenAI-style function calling via the configured LLM.
"""
from typing import List, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from llm import get_chat_llm
from tool_router import TOOLS

_base_llm = None


def _get_base_llm():
    global _base_llm
    if _base_llm is None:
        _base_llm = get_chat_llm()
    return _base_llm


def plan_step(
    messages: List[BaseMessage], tools: Sequence[BaseTool] = TOOLS
) -> BaseMessage:
    """
    Given the running message history and the set of tools available for
    this call, returns the LLM's next message — either a request to call
    one or more tools, or a direct answer.
    """
    return _get_base_llm().bind_tools(list(tools)).invoke(messages)
