"""
Agent orchestration loop: multi-step planning + tool execution + persistent
memory, on top of the document-analysis RAG pipeline.
"""
from typing import Dict, Any, List

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from config import CONFIG
from memory import get_memory
from planner import plan_step
from tool_router import (
    DEFAULT_COLLECTION,
    TOOLS,
    TOOLS_BY_NAME,
    make_tools,
    search_knowledge_base_raw,
    format_search_results,
)

MAX_ITERATIONS = CONFIG.get("agent", {}).get("max_tool_iterations", 4)

SYSTEM_PROMPT = (
    "You are the assistant for a document analysis tool. Users upload "
    "documents and ask questions about them. Use the search_knowledge_base "
    "tool to find relevant passages before answering questions about "
    "document content — do not answer from general knowledge. Use "
    "list_documents or get_document_stats for questions about what's in "
    "the knowledge base. Use generate_chart when the user asks for a "
    "chart, graph, or visualization. Cite which source each fact came from "
    "when you can, and say plainly when the knowledge base doesn't have "
    "the answer."
)


def _history_to_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    role_map = {"user": HumanMessage, "assistant": AIMessage}
    messages: List[BaseMessage] = []
    for turn in history:
        cls = role_map.get(turn["role"])
        if cls:
            messages.append(cls(content=turn["content"]))
    return messages


def _execute_tool_call(
    call: Dict[str, Any],
    tools_by_name: Dict[str, Any],
    collection_name: str,
    sources: List[str],
) -> str:
    name = call["name"]
    args = call.get("args", {})

    if name == "search_knowledge_base":
        docs = search_knowledge_base_raw(
            args.get("query", ""), collection_name=collection_name
        )
        sources.extend(doc.page_content for doc in docs)
        return format_search_results(docs)

    tool_fn = tools_by_name.get(name)
    if tool_fn is None:
        return f"Unknown tool: {name}"
    return tool_fn.invoke(args)


def run_agent(
    query: str,
    session_id: str = "default",
    collection_name: str = DEFAULT_COLLECTION,
) -> Dict[str, Any]:
    """
    Runs one turn of the agent: loads conversation history, lets the LLM
    plan and call tools (RAG search, doc stats, chart generation) across up
    to MAX_ITERATIONS steps, then persists and returns the final answer.

    `collection_name` scopes which knowledge base the tools operate on —
    defaults to the single shared collection used for normal CLI/API use;
    the public demo passes a per-visitor collection instead so visitors
    don't see each other's uploads.
    """
    memory = get_memory()
    history = memory.get_history(session_id)

    if collection_name == DEFAULT_COLLECTION:
        tools, tools_by_name = TOOLS, TOOLS_BY_NAME
    else:
        tools, tools_by_name = make_tools(collection_name)

    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages += _history_to_messages(history)
    messages.append(HumanMessage(content=query))

    tool_trace: List[Dict[str, Any]] = []
    sources: List[str] = []

    for _ in range(MAX_ITERATIONS):
        ai_message = plan_step(messages, tools=tools)
        messages.append(ai_message)

        tool_calls = getattr(ai_message, "tool_calls", None)
        if not tool_calls:
            answer = ai_message.content
            memory.add_message(session_id, "user", query)
            memory.add_message(session_id, "assistant", answer)
            return {"answer": answer, "tool_calls": tool_trace, "sources": sources}

        for call in tool_calls:
            result = _execute_tool_call(call, tools_by_name, collection_name, sources)
            tool_trace.append(
                {"tool": call["name"], "args": call.get("args", {}), "result": result}
            )
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    # Ran out of iterations — force a final answer without further tool calls.
    messages.append(
        HumanMessage(
            content="Please give your best answer now based on what you've "
            "found, without calling any more tools."
        )
    )
    final = plan_step(messages, tools=tools).content
    memory.add_message(session_id, "user", query)
    memory.add_message(session_id, "assistant", final)
    return {"answer": final, "tool_calls": tool_trace, "sources": sources}
