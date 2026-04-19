from app.agents.langgraph_flow import build_langgraph

graph = build_langgraph()


def format_inventory_answer(question, tool_results):
    inventory_result = tool_results.get("inventory", {})
    display_df = inventory_result.get("display_df")

    if display_df is None or display_df.empty:
        return "No inventory data is available."

    q = question.lower().strip()

    # Case 1: Low stock / restock style question
    if any(word in q for word in ["restock", "threshold", "low stock", "urgent", "immediately"]):
        if {"quantity", "reorder_threshold"}.issubset(display_df.columns):
            display_df = display_df[display_df["quantity"] < display_df["reorder_threshold"]].copy()

        if display_df.empty:
            return "No inventory items need immediate restocking."

        if "quantity" in display_df.columns:
            display_df = display_df.sort_values(by="quantity", ascending=True)

        lines = []
        if "name" in display_df.columns:
            item_names = display_df["name"].tolist()
            lines.append(
                "You should restock these items immediately: "
                + ", ".join(item_names) + "."
            )

        if {"name", "quantity", "reorder_threshold"}.issubset(display_df.columns):
            lines.append("\nMost urgent items:")
            for _, row in display_df.head(5).iterrows():
                lines.append(
                    f"- {row['name']}: current stock {row['quantity']}, reorder threshold {row['reorder_threshold']}"
                )

        return "\n".join(lines)

    # Case 2: Supplier question
    if "supplier" in q:
        if {"name", "supplier"}.issubset(display_df.columns):
            lines = ["Here are the warehouse products and their suppliers:"]
            for _, row in display_df.iterrows():
                lines.append(f"- {row['name']}: supplier {row['supplier']}")
            return "\n".join(lines)

    # Case 3: Location / storage question
    if any(word in q for word in ["where", "location", "stored"]):
        if {"name", "location"}.issubset(display_df.columns):
            lines = ["Here are the warehouse products and their locations:"]
            for _, row in display_df.iterrows():
                lines.append(f"- {row['name']}: stored at {row['location']}")
            return "\n".join(lines)

    # Case 4: General product list question
    if any(word in q for word in ["what products", "products are there", "inventory list", "all products", "items are there"]):
        if "name" in display_df.columns:
            product_names = display_df["name"].tolist()
            return "The products currently available in the warehouse are: " + ", ".join(product_names) + "."

    # Default generic inventory answer
    if "name" in display_df.columns:
        product_names = display_df["name"].tolist()
        return "Here is the current warehouse inventory: " + ", ".join(product_names) + "."

    return "Inventory results are available in the table below."


def format_shipment_answer(tool_results):
    shipment_result = tool_results.get("shipment", {})
    display_df = shipment_result.get("display_df")

    if display_df is None or display_df.empty:
        return "There are no delayed shipments right now."

    lines = []
    lines.append("These shipments are currently delayed:")

    required_cols = {"shipment_id", "supplier", "expected_date", "notes"}
    if required_cols.issubset(display_df.columns):
        for _, row in display_df.iterrows():
            lines.append(
                f"- {row['shipment_id']} from {row['supplier']}, expected on {row['expected_date']}. Reason: {row['notes']}"
            )
    else:
        lines.append("- Delayed shipment information is available in the table below.")

    return "\n".join(lines)


def format_document_answer(tool_results):
    document_result = tool_results.get("document", {})
    text_output = document_result.get("text_output", "")

    if not text_output:
        return "I could not find a matching warehouse document answer."

    lines = text_output.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if (
            line.startswith("FILE:")
            or line.startswith("RELEVANCE")
            or line.startswith("BEST MATCHING CONTENT")
            or not line
        ):
            continue

        cleaned_lines.append(line)

    if not cleaned_lines:
        return "Relevant warehouse document found, but unable to extract clean summary."

    final_text = " ".join(cleaned_lines)

    return f"Warehouse process summary:\n\n{final_text}"


def format_decision_answer(tool_results):
    decision_result = tool_results.get("decision", {})
    text_output = decision_result.get("text_output", "").strip()
    display_df = decision_result.get("display_df")

    if text_output:
        return text_output

    if display_df is not None and not display_df.empty:
        return "I found decision-support results in the structured table below."

    return "I could not generate a decision recommendation."


def polish_answer(question, selected_tools, tool_results, final_answer):
    final_answer = (final_answer or "").strip()

    raw_prefixes = [
        "INVENTORY RESULT:",
        "SHIPMENT RESULT:",
        "DOCUMENT RESULT:",
        "DECISION RESULT:",
    ]

    is_raw = any(final_answer.upper().startswith(prefix) for prefix in raw_prefixes)

    if not final_answer or is_raw:
        polished_parts = []

        if "inventory" in selected_tools:
            polished_parts.append(format_inventory_answer(question, tool_results))

        if "shipment" in selected_tools:
            polished_parts.append(format_shipment_answer(tool_results))

        if "document" in selected_tools:
            polished_parts.append(format_document_answer(tool_results))

        if "decision" in selected_tools:
            polished_parts.append(format_decision_answer(tool_results))

        polished_parts = [part for part in polished_parts if part.strip()]

        if polished_parts:
            return "\n\n".join(polished_parts)

    return final_answer


def run_langgraph_question(question: str):
    result = graph.invoke({"user_question": question})

    selected_tools = result.get("selected_tools", [])
    tool_results = result.get("tool_results", {})
    final_answer = result.get("final_answer", "")
    resolved_question = result.get("resolved_question", question)

    final_answer = polish_answer(resolved_question, selected_tools, tool_results, final_answer)

    return final_answer, selected_tools, tool_results, resolved_question