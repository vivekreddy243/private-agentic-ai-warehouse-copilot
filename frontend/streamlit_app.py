import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama

DB_PATH = "data/warehouse.db"
DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Private Agentic AI Copilot", layout="wide")
st.title("Private Agentic AI Copilot for Warehouse Operations")
st.caption("Electronics E-commerce Fulfillment Warehouse")

llm = ChatOllama(model="gemma3:1b")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

if "last_inventory_result_df" not in st.session_state:
    st.session_state.last_inventory_result_df = None


# -----------------------------
# DATABASE HELPERS
# -----------------------------
def get_products_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM products", conn)
    conn.close()
    return df


def get_shipments_df():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM shipments", conn)
    conn.close()
    return df


def get_sales_df():
    sales_data = [
        {"product": "Wireless Mouse", "category": "Electronics", "month": "Jan", "units_sold": 32},
        {"product": "Wireless Mouse", "category": "Electronics", "month": "Feb", "units_sold": 28},
        {"product": "Wireless Mouse", "category": "Electronics", "month": "Mar", "units_sold": 35},
        {"product": "Mechanical Keyboard", "category": "Electronics", "month": "Jan", "units_sold": 20},
        {"product": "Mechanical Keyboard", "category": "Electronics", "month": "Feb", "units_sold": 18},
        {"product": "Mechanical Keyboard", "category": "Electronics", "month": "Mar", "units_sold": 22},
        {"product": "Portable SSD", "category": "Electronics", "month": "Jan", "units_sold": 12},
        {"product": "Portable SSD", "category": "Electronics", "month": "Feb", "units_sold": 15},
        {"product": "Portable SSD", "category": "Electronics", "month": "Mar", "units_sold": 19},
        {"product": "Thermal Printer", "category": "Warehouse Supplies", "month": "Jan", "units_sold": 4},
        {"product": "Thermal Printer", "category": "Warehouse Supplies", "month": "Feb", "units_sold": 3},
        {"product": "Thermal Printer", "category": "Warehouse Supplies", "month": "Mar", "units_sold": 5},
        {"product": "Barcode Scanner", "category": "Warehouse Supplies", "month": "Jan", "units_sold": 5},
        {"product": "Barcode Scanner", "category": "Warehouse Supplies", "month": "Feb", "units_sold": 6},
        {"product": "Barcode Scanner", "category": "Warehouse Supplies", "month": "Mar", "units_sold": 7},
    ]
    return pd.DataFrame(sales_data)


# -----------------------------
# FOLLOW-UP MEMORY HELPER
# -----------------------------
def resolve_followup_question(question: str) -> str:
    q = question.lower().strip()
    prev = st.session_state.last_user_question.lower().strip()

    if not prev:
        return question

    followup_words = ["them", "those", "that", "those items", "those shipments", "it"]

    if any(word in q for word in followup_words):
        if "who supplies" in q and ("low stock" in prev or "reorder threshold" in prev):
            return "Who supplies the items that are below reorder threshold?"
        if ("where are" in q or "where is" in q) and ("low stock" in prev or "reorder threshold" in prev):
            return "Where are the items that are below reorder threshold stored?"
        if "what should we do" in q and "delayed" in prev:
            return "What should we do if a supplier shipment is delayed?"
        return f"{question} (follow-up to: {st.session_state.last_user_question})"

    return question


# -----------------------------
# TOOLS
# -----------------------------
def inventory_tool(question: str):
    df = get_products_df()
    q = question.lower().strip()

    if "low stock" in q or "below reorder threshold" in q or "restock" in q:
        result = df[df["quantity"] < df["reorder_threshold"]][
            ["sku", "name", "quantity", "reorder_threshold", "location", "supplier"]
        ]
        st.session_state.last_inventory_result_df = result.copy() if not result.empty else None

        if result.empty:
            return {
                "name": "inventory",
                "display_df": None,
                "text_output": "No items are currently below reorder threshold."
            }
        return {
            "name": "inventory",
            "display_df": result,
            "text_output": result.to_string(index=False)
        }

    elif "where is" in q or "where are" in q:
        if ("them" in q or "those" in q) and st.session_state.last_inventory_result_df is not None:
            result = st.session_state.last_inventory_result_df[["sku", "name", "location", "quantity", "supplier"]]
        else:
            search_term = (
                q.replace("where is", "")
                 .replace("where are", "")
                 .replace("stored", "")
                 .replace("?", "")
                 .strip()
            )
            result = df[df["name"].str.lower().str.contains(search_term, na=False)][
                ["sku", "name", "location", "quantity", "supplier"]
            ]

        if result.empty:
            return {
                "name": "inventory",
                "display_df": None,
                "text_output": "No matching product location found."
            }
        return {
            "name": "inventory",
            "display_df": result,
            "text_output": result.to_string(index=False)
        }

    elif "supplier" in q or "who supplies" in q:
        if ("them" in q or "those" in q) and st.session_state.last_inventory_result_df is not None:
            base_df = st.session_state.last_inventory_result_df.copy()
            available_cols = [c for c in ["sku", "name", "supplier", "location", "quantity"] if c in base_df.columns]
            result = base_df[available_cols]
        else:
            result = df[["sku", "name", "supplier", "location", "quantity"]]

        if result.empty:
            return {
                "name": "inventory",
                "display_df": None,
                "text_output": "No supplier information found for the requested items."
            }
        return {
            "name": "inventory",
            "display_df": result,
            "text_output": result.to_string(index=False)
        }

    else:
        result = df[["sku", "name", "category", "quantity", "location", "supplier"]]
        return {
            "name": "inventory",
            "display_df": result,
            "text_output": result.to_string(index=False)
        }


def shipment_tool(question: str):
    df = get_shipments_df()
    q = question.lower().strip()

    if "delayed" in q or "shipment" in q or "delivery" in q:
        result = df[df["status"].str.lower() == "delayed"][
            ["shipment_id", "supplier", "status", "expected_date", "notes"]
        ]
        if result.empty:
            return {
                "name": "shipment",
                "display_df": None,
                "text_output": "No delayed shipments found."
            }
        return {
            "name": "shipment",
            "display_df": result,
            "text_output": result.to_string(index=False)
        }

    result = df[["shipment_id", "supplier", "status", "expected_date", "notes"]]
    return {
        "name": "shipment",
        "display_df": result,
        "text_output": result.to_string(index=False)
    }


def document_tool(question: str):
    q = question.lower().strip()
    matched_docs = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if any(word in content.lower() for word in q.split()):
                    matched_docs.append({
                        "filename": file_path.name,
                        "content": content[:3000]
                    })
            except Exception:
                continue

    if not matched_docs:
        return {
            "name": "document",
            "display_df": None,
            "text_output": "No relevant document content found."
        }

    df = pd.DataFrame(
        [{"filename": d["filename"], "preview": d["content"][:200]} for d in matched_docs]
    )

    combined_text = "\n\n".join(
        [f"FILE: {d['filename']}\nCONTENT:\n{d['content']}" for d in matched_docs]
    )

    return {
        "name": "document",
        "display_df": df,
        "text_output": combined_text
    }


# -----------------------------
# DETERMINISTIC ANSWER HELPERS
# -----------------------------
def generate_inventory_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()

    if df is None or df.empty:
        return "No matching inventory information found."

    if "low stock" in q or "below reorder threshold" in q or "restock" in q:
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"{row['name']} is below threshold with quantity {row['quantity']} against reorder threshold {row['reorder_threshold']}. "
                f"It is stored at {row['location']} and supplied by {row['supplier']}."
            )
        return "The following items need attention: " + " ".join(lines)

    if "who supplies" in q or "supplier" in q:
        lines = []
        for _, row in df.iterrows():
            supplier = row["supplier"] if "supplier" in df.columns else "unknown supplier"
            lines.append(f"{row['name']} is supplied by {supplier}.")
        return "Supplier information: " + " ".join(lines)

    if "where is" in q or "where are" in q or "stored" in q or "location" in q:
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"{row['name']} is stored at {row['location']}. Current quantity is {row['quantity']}."
            )
        return "Storage details: " + " ".join(lines)

    return ""


def generate_shipment_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()

    if df is None or df.empty:
        return "No matching shipment information found."

    if "delayed" in q or "shipment" in q or "delivery" in q:
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"Shipment {row['shipment_id']} from {row['supplier']} is {row['status']} with expected date {row['expected_date']}. "
                f"Notes: {row['notes']}."
            )
        return "Delayed shipment details: " + " ".join(lines)

    return ""


def should_use_deterministic_answer(resolved_question: str, selected_tools: list) -> bool:
    q = resolved_question.lower()

    # Process/policy/SOP style questions should go to the document + LLM path
    document_priority_words = [
        "sop", "policy", "process", "document", "instruction",
        "guideline", "damaged", "return", "returns"
    ]
    if any(word in q for word in document_priority_words):
        return False

    deterministic_patterns = [
        "low stock",
        "below reorder threshold",
        "restock",
        "who supplies",
        "supplier",
        "where is",
        "where are",
        "stored",
        "location",
        "which shipments are delayed",
        "show delayed shipments",
        "delayed shipment",
    ]

    return any(pattern in q for pattern in deterministic_patterns)


# -----------------------------
# SIMPLE PLANNER
# -----------------------------
def planner(question: str):
    q = question.lower()

    inventory_words = ["stock", "threshold", "restock", "where is", "where are", "supplier", "location", "stored"]
    shipment_words = ["shipment", "delayed", "delivery", "expected date"]
    document_words = ["sop", "policy", "process", "document", "instruction", "guideline", "damaged", "return", "returns"]

    # Document-first routing for pure process/policy/doc questions
    if any(word in q for word in document_words) and not any(word in q for word in inventory_words + shipment_words):
        return ["document"]

    selected = []

    if any(word in q for word in inventory_words):
        selected.append("inventory")

    if any(word in q for word in shipment_words):
        selected.append("shipment")

    if any(word in q for word in document_words):
        selected.append("document")

    if not selected:
        selected = ["inventory"]

    return selected


def build_chat_history_text():
    if not st.session_state.chat_history:
        return "No previous conversation."

    history_lines = []
    for msg in st.session_state.chat_history[-6:]:
        history_lines.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(history_lines)


def run_agentic_flow(question: str):
    resolved_question = resolve_followup_question(question)
    selected_tools = planner(resolved_question)
    tool_results = {}

    if "inventory" in selected_tools:
        tool_results["inventory"] = inventory_tool(resolved_question)

    if "shipment" in selected_tools:
        tool_results["shipment"] = shipment_tool(resolved_question)

    if "document" in selected_tools:
        tool_results["document"] = document_tool(resolved_question)

    # Deterministic answers for structured operational questions
    if should_use_deterministic_answer(resolved_question, selected_tools):
        deterministic_parts = []

        if "inventory" in tool_results and tool_results["inventory"]["display_df"] is not None:
            inv_answer = generate_inventory_answer(
                resolved_question, tool_results["inventory"]["display_df"]
            )
            if inv_answer:
                deterministic_parts.append(inv_answer)

        if "shipment" in tool_results and tool_results["shipment"]["display_df"] is not None:
            ship_answer = generate_shipment_answer(
                resolved_question, tool_results["shipment"]["display_df"]
            )
            if ship_answer:
                deterministic_parts.append(ship_answer)

        if deterministic_parts:
            final_answer = " ".join(deterministic_parts)
            used_sources = ", ".join(selected_tools)
            final_answer += f" Data source used: {used_sources}."
            return final_answer, selected_tools, tool_results, resolved_question

    # LLM path for document/mixed/explanatory queries
    combined_context = "\n\n".join(
        [
            f"{tool_name.upper()} DATA:\n{tool_result['text_output']}"
            for tool_name, tool_result in tool_results.items()
        ]
    )

    chat_history_text = build_chat_history_text()

    prompt = f"""
You are a private warehouse AI copilot for an electronics e-commerce fulfillment warehouse.

Use only the tool results below to answer the user's question.
If the answer is not available, clearly say that it is not available.
Use exact values from the data whenever possible.
Do not make vague statements if exact rows are available.

CHAT HISTORY:
{chat_history_text}

RESOLVED QUESTION:
{resolved_question}

{combined_context}

Give a short, clear, business-friendly answer.
If a product location, supplier, quantity, shipment status, or document instruction is available, mention the exact value.
Also mention which data source you used: inventory, shipment, document, or a combination.
"""

    response = llm.invoke(prompt)
    return response.content, selected_tools, tool_results, resolved_question


# -----------------------------
# ANALYTICS + ALERTS + RECOMMENDATIONS
# -----------------------------
def build_dashboard_metrics():
    products_df = get_products_df()
    shipments_df = get_shipments_df()
    sales_df = get_sales_df()

    total_products = len(products_df)
    low_stock_df = products_df[products_df["quantity"] < products_df["reorder_threshold"]]
    low_stock_count = len(low_stock_df)

    delayed_shipments_df = shipments_df[shipments_df["status"].str.lower() == "delayed"]
    delayed_shipments_count = len(delayed_shipments_df)

    total_suppliers = products_df["supplier"].nunique()

    category_summary = (
        products_df.groupby("category", as_index=False)["quantity"]
        .sum()
        .sort_values(by="quantity", ascending=False)
    )

    supplier_summary = (
        products_df.groupby("supplier", as_index=False)["quantity"]
        .sum()
        .sort_values(by="quantity", ascending=False)
    )

    monthly_sales = sales_df.groupby("month", as_index=False)["units_sold"].sum()
    product_sales = (
        sales_df.groupby("product", as_index=False)["units_sold"]
        .sum()
        .sort_values(by="units_sold", ascending=False)
    )

    return {
        "products_df": products_df,
        "shipments_df": shipments_df,
        "sales_df": sales_df,
        "low_stock_df": low_stock_df,
        "delayed_shipments_df": delayed_shipments_df,
        "total_products": total_products,
        "low_stock_count": low_stock_count,
        "delayed_shipments_count": delayed_shipments_count,
        "total_suppliers": total_suppliers,
        "category_summary": category_summary,
        "supplier_summary": supplier_summary,
        "monthly_sales": monthly_sales,
        "product_sales": product_sales,
    }


def build_alerts_and_recommendations(dashboard):
    alerts = []
    recommendations = []

    low_stock_df = dashboard["low_stock_df"]
    delayed_shipments_df = dashboard["delayed_shipments_df"]

    if not low_stock_df.empty:
        alerts.append(f"{len(low_stock_df)} items are below reorder threshold.")

        critical_items = low_stock_df.sort_values(by="quantity").head(3)
        critical_names = ", ".join(critical_items["name"].tolist())
        alerts.append(f"Critical low-stock items: {critical_names}.")

        recommendations.append(
            "Prioritize restocking the lowest-quantity items first: "
            + ", ".join(critical_items["name"].tolist()) + "."
        )

    if not delayed_shipments_df.empty:
        alerts.append(f"{len(delayed_shipments_df)} shipments are currently delayed.")
        delayed_suppliers = ", ".join(delayed_shipments_df["supplier"].unique().tolist())
        recommendations.append(
            f"Follow up with delayed shipment suppliers: {delayed_suppliers}."
        )

    if not low_stock_df.empty and not delayed_shipments_df.empty:
        recommendations.append(
            "Review whether delayed supplier shipments are affecting replenishment for low-stock products."
        )

    if dashboard["product_sales"].shape[0] > 0:
        top_sales = dashboard["product_sales"].head(3)
        recommendations.append(
            "Top-selling products to monitor closely: "
            + ", ".join(top_sales["product"].tolist()) + "."
        )

    if not alerts:
        alerts.append("No major operational alerts right now.")

    if not recommendations:
        recommendations.append("No urgent recommendations at the moment.")

    return alerts, recommendations


# -----------------------------
# UI
# -----------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Inventory Admin",
        "AI Copilot",
        "Warehouse Knowledge Base",
        "Analytics Dashboard",
        "Alerts & Recommendations",
        "Project Info",
    ]
)

with tab1:
    st.subheader("Inventory Admin")
    st.write("View inventory and add new warehouse products.")

    df = get_products_df()
    st.dataframe(df, width="stretch")

    st.subheader("Add New Product")

    with st.form("add_product_form"):
        sku = st.text_input("SKU")
        name = st.text_input("Product Name")
        category = st.selectbox(
            "Category",
            ["Electronics", "Accessories", "Warehouse Supplies"]
        )
        quantity = st.number_input("Quantity", min_value=0, value=0)
        threshold = st.number_input("Reorder Threshold", min_value=0, value=10)
        location = st.text_input("Location")
        supplier = st.text_input("Supplier")

        submitted = st.form_submit_button("Add Product")

        if submitted:
            try:
                conn = sqlite3.connect(DB_PATH)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO products
                    (sku, name, category, quantity, reorder_threshold, location, supplier)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (sku, name, category, quantity, threshold, location, supplier))
                conn.commit()
                conn.close()
                st.success("Product added successfully.")
                st.rerun()
            except sqlite3.IntegrityError:
                st.error("SKU already exists. Please use a unique SKU.")

with tab2:
    st.subheader("AI Copilot")
    st.write("Ask questions about stock, suppliers, locations, shipments, or warehouse documents.")

    with st.expander("Sample Questions"):
        st.write("- Which items are below reorder threshold?")
        st.write("- Which shipments are delayed?")
        st.write("- Where is Wireless Mouse stored?")
        st.write("- What is the process for damaged goods?")
        st.write("- Which items need restocking and what is the delayed shipment process?")

    if st.session_state.chat_history:
        st.subheader("Recent Conversation")
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**Copilot:** {msg['content']}")

    question = st.text_input("Ask a warehouse question")

    col1, col2 = st.columns([1, 1])

    with col1:
        submit_clicked = st.button("Submit Question")

    with col2:
        clear_clicked = st.button("Clear Chat History")

    if clear_clicked:
        st.session_state.chat_history = []
        st.session_state.last_user_question = ""
        st.session_state.last_inventory_result_df = None
        st.success("Chat history cleared.")
        st.rerun()

    if submit_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    final_answer, selected_tools, tool_results, resolved_question = run_agentic_flow(question)

                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
                    st.session_state.last_user_question = question

                    st.success("Answer:")
                    st.write(final_answer)

                    st.info(f"Resolved question: {resolved_question}")
                    st.info(f"Planner selected tools: {', '.join(selected_tools)}")

                    st.subheader("Structured Results")
                    shown_any_table = False

                    for tool_name, tool_result in tool_results.items():
                        if tool_result["display_df"] is not None:
                            st.markdown(f"### {tool_name.title()} Results")
                            st.dataframe(tool_result["display_df"], width="stretch")
                            shown_any_table = True

                    if not shown_any_table:
                        st.write("No tabular results available for this query.")

                    with st.expander("See raw tool outputs"):
                        for tool_name, tool_result in tool_results.items():
                            st.markdown(f"### {tool_name.title()} Tool Output")
                            st.text(tool_result["text_output"])

                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.subheader("Warehouse Knowledge Base")
    st.write("Upload .txt files for SOPs, warehouse instructions, return policies, or supplier notes.")

    uploaded_file = st.file_uploader("Upload a text document", type=["txt"])

    if uploaded_file is not None:
        save_path = DOCS_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

    st.subheader("Available Documents")
    files = [f.name for f in DOCS_DIR.iterdir() if f.is_file()]
    if files:
        for name in files:
            st.write(f"- {name}")
    else:
        st.info("No documents uploaded yet.")

with tab4:
    st.subheader("Warehouse Analytics Dashboard")
    st.write("Operational overview of stock, suppliers, shipment risk, and sales trend.")

    dashboard = build_dashboard_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Products", dashboard["total_products"])
    col2.metric("Low Stock Items", dashboard["low_stock_count"])
    col3.metric("Delayed Shipments", dashboard["delayed_shipments_count"])
    col4.metric("Total Suppliers", dashboard["total_suppliers"])

    st.subheader("Monthly Sales Trend")
    st.line_chart(dashboard["monthly_sales"].set_index("month"))

    st.subheader("Product Sales Summary")
    st.bar_chart(dashboard["product_sales"].set_index("product"))

    st.subheader("Category-wise Stock Summary")
    st.dataframe(dashboard["category_summary"], width="stretch")

    st.subheader("Supplier-wise Quantity Summary")
    st.dataframe(dashboard["supplier_summary"], width="stretch")

    st.subheader("Low Stock Products")
    if dashboard["low_stock_df"].empty:
        st.success("No low stock items.")
    else:
        st.dataframe(
            dashboard["low_stock_df"][["sku", "name", "quantity", "reorder_threshold", "location", "supplier"]],
            width="stretch"
        )

    st.subheader("Delayed Shipments")
    if dashboard["delayed_shipments_df"].empty:
        st.success("No delayed shipments.")
    else:
        st.dataframe(
            dashboard["delayed_shipments_df"][["shipment_id", "supplier", "status", "expected_date", "notes"]],
            width="stretch"
        )

with tab5:
    st.subheader("Smart Alerts & Recommendations")
    st.write("Proactive operational signals based on current inventory, shipment, and sales conditions.")

    dashboard = build_dashboard_metrics()
    alerts, recommendations = build_alerts_and_recommendations(dashboard)

    st.markdown("### Alerts")
    for alert in alerts:
        st.warning(alert)

    st.markdown("### Recommendations")
    for rec in recommendations:
        st.success(rec)

with tab6:
    st.subheader("Project Info")
    st.write("Overview of the project, its purpose, architecture, and technology stack.")

    st.markdown("""
### Project Title
**Private Agentic AI Copilot for Warehouse Operations**

### Problem Statement
Warehouse operations data is often split across inventory tables, shipment records, and operational documents. Staff must manually search across multiple sources to answer common operational questions.

### Proposed Solution
This project builds a private local AI copilot that:
- reads structured warehouse data
- retrieves warehouse knowledge documents
- uses a planner to select the right tools
- returns grounded responses through a local LLM

### Key Features
- Inventory admin module
- AI copilot with follow-up memory
- Warehouse knowledge base
- Analytics dashboard
- Smart alerts and recommendations
- Private local LLM using Ollama

### Agentic AI Aspect
The system is agentic because it does not answer blindly. It first analyzes the question, selects relevant tools such as inventory, shipment, or document retrieval, gathers the required data, and then generates the final response.

### Privacy Aspect
- local database
- local documents
- local LLM
- no cloud API required

### Tech Stack
- Python
- Streamlit
- SQLite
- Ollama
- LangChain-Ollama
- Pandas
""")