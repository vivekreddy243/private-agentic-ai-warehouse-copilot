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


# -----------------------------
# TOOLS
# -----------------------------
def inventory_tool(question: str):
    df = get_products_df()
    q = question.lower().strip()

    if "low stock" in q or "below threshold" in q or "restock" in q:
        result = df[df["quantity"] < df["reorder_threshold"]][
            ["sku", "name", "quantity", "reorder_threshold", "location", "supplier"]
        ]
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

    elif "where is" in q:
        search_term = q.replace("where is", "").replace("stored", "").replace("?", "").strip()
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

    elif "supplier" in q:
        result = df[["sku", "name", "supplier", "location", "quantity"]]
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
# SIMPLE PLANNER
# -----------------------------
def planner(question: str):
    q = question.lower()

    inventory_words = ["stock", "threshold", "restock", "where is", "supplier", "location", "stored"]
    shipment_words = ["shipment", "delayed", "delivery", "expected date"]
    document_words = ["sop", "policy", "process", "document", "instruction", "guideline", "damaged", "return"]

    selected = []

    if any(word in q for word in inventory_words):
        selected.append("inventory")

    if any(word in q for word in shipment_words):
        selected.append("shipment")

    if any(word in q for word in document_words):
        selected.append("document")

    if not selected:
        selected = ["inventory", "shipment", "document"]

    return selected


def run_agentic_flow(question: str):
    selected_tools = planner(question)
    tool_results = {}

    if "inventory" in selected_tools:
        tool_results["inventory"] = inventory_tool(question)

    if "shipment" in selected_tools:
        tool_results["shipment"] = shipment_tool(question)

    if "document" in selected_tools:
        tool_results["document"] = document_tool(question)

    combined_context = "\n\n".join(
        [
            f"{tool_name.upper()} DATA:\n{tool_result['text_output']}"
            for tool_name, tool_result in tool_results.items()
        ]
    )

    prompt = f"""
You are a private warehouse AI copilot for an electronics e-commerce fulfillment warehouse.

Use only the tool results below to answer the user's question.
If the answer is not available, clearly say that it is not available.
Use exact values from the data whenever possible.

{combined_context}

USER QUESTION:
{question}

Give a short, clear, business-friendly answer.
If a product location, supplier, quantity, shipment status, or document instruction is available, mention the exact value.
Also mention which data source you used: inventory, shipment, document, or a combination.
"""

    response = llm.invoke(prompt)
    return response.content, selected_tools, tool_results


# -----------------------------
# UI
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Admin Panel", "Warehouse Chat", "Documents"])

with tab1:
    st.subheader("Current Inventory")
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
    st.subheader("Warehouse Chat")
    st.write("Ask questions about stock, suppliers, locations, shipments, or warehouse documents.")

    question = st.text_input("Ask a warehouse question")

    if st.button("Submit Question"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    final_answer, selected_tools, tool_results = run_agentic_flow(question)

                    st.success("Answer:")
                    st.write(final_answer)

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
    st.subheader("Upload Warehouse Documents")
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