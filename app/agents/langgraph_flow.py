from langgraph.graph import StateGraph, END
from app.agents.state import WarehouseAgentState
from app.agents.tool_helpers import (
    inventory_tool_logic,
    shipment_tool_logic,
    document_tool_logic,
    decision_tool_logic,
)


def query_understanding_node(state: WarehouseAgentState) -> WarehouseAgentState:
    question = state.get("user_question", "").strip()
    state["resolved_question"] = question
    return state


def planner_node(state: WarehouseAgentState) -> WarehouseAgentState:
    q = state.get("resolved_question", "").lower()

    inventory_words = [
        "stock", "threshold", "restock", "where is", "where are",
        "supplier", "location", "stored", "inventory", "quantity", "low stock"
    ]
    shipment_words = [
        "shipment", "shipments", "delayed", "delivery", "expected date", "supplier delay"
    ]
    document_words = [
        "sop", "policy", "process", "document", "instruction",
        "guideline", "damaged", "return", "returns"
    ]
    decision_words = [
        "recommend",
        "priority",
        "restock first",
        "which should be restocked first",
        "which items should be restocked first",
        "top items",
        "restocked first",
        "affected by delayed shipments",
        "what should i do",
        "urgent",
        "immediately"
    ]

    selected = []

    if any(word in q for word in inventory_words):
        selected.append("inventory")

    if any(word in q for word in shipment_words):
        selected.append("shipment")

    if any(word in q for word in document_words):
        selected.append("document")

    if any(word in q for word in decision_words):
        selected.append("decision")

    # remove duplicates, keep order
    selected = list(dict.fromkeys(selected))

    if not selected:
        selected = ["inventory"]

    state["selected_tools"] = selected
    return state


def tool_execution_node(state: WarehouseAgentState) -> WarehouseAgentState:
    selected_tools = state.get("selected_tools", [])
    resolved_question = state.get("resolved_question", "")
    tool_results = state.get("tool_results", {})

    if "inventory" in selected_tools:
        tool_results["inventory"] = inventory_tool_logic(resolved_question)

    if "shipment" in selected_tools:
        tool_results["shipment"] = shipment_tool_logic(resolved_question)

    if "document" in selected_tools:
        tool_results["document"] = document_tool_logic(resolved_question)

    if "decision" in selected_tools:
        tool_results["decision"] = decision_tool_logic()

    state["tool_results"] = tool_results
    return state


def response_node(state: WarehouseAgentState) -> WarehouseAgentState:
    selected_tools = state.get("selected_tools", [])
    tool_results = state.get("tool_results", {})

    answer_parts = []

    for tool_name in selected_tools:
        if tool_name in tool_results:
            text_output = tool_results[tool_name].get("text_output", "")
            if text_output:
                answer_parts.append(f"{tool_name.upper()} RESULT:\n{text_output}")

    if answer_parts:
        state["final_answer"] = "\n\n".join(answer_parts)
    else:
        state["final_answer"] = f"LangGraph response generated using: {', '.join(selected_tools)}"

    return state


def build_langgraph():
    graph = StateGraph(WarehouseAgentState)

    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("planner", planner_node)
    graph.add_node("tool_execution", tool_execution_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "planner")
    graph.add_edge("planner", "tool_execution")
    graph.add_edge("tool_execution", "response")
    graph.add_edge("response", END)

    return graph.compile()