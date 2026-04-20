from app.agents.tool_helpers import document_tool_logic


def summarize_documents(tool_results):
    document_result = tool_results.get("document", {})
    text_output = document_result.get("text_output", "")

    if not text_output:
        return "I could not find a matching warehouse document answer."

    cleaned_lines = []
    for line in text_output.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("BEST MATCHING CONTENT:"):
            content = line.split(":", 1)[1].strip()
            if content:
                cleaned_lines.append(content)
            continue
        if line.startswith("FILE:") or line.startswith("RELEVANCE"):
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return "Relevant warehouse document found, but I could not extract a clean summary."

    best_text = " ".join(cleaned_lines[:3]).strip()
    return f"Warehouse process summary:\n\n{best_text}"


def run_langgraph_question(question: str):
    tool_results = {"document": document_tool_logic(question)}
    selected_tools = ["document"]
    final_answer = summarize_documents(tool_results)
    resolved_question = question
    return final_answer, selected_tools, tool_results, resolved_question
