from typing import TypedDict, List, Dict, Optional, Any


class WarehouseAgentState(TypedDict, total=False):
    user_question: str
    resolved_question: str
    selected_tools: List[str]
    tool_results: Dict[str, Dict[str, Any]]
    final_answer: str
    chat_history_text: str
    alerts: List[str]
    recommendations: List[str]