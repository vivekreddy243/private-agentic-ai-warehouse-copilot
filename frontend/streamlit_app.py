import sys
from pathlib import Path
import sqlite3

# Make project root importable so "app.agents..." works in Streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from app.agents.langgraph_runner import run_langgraph_question

DB_PATH = "data/warehouse.db"
DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Private Agentic AI Copilot", layout="wide")
st.title("Private Agentic AI Copilot for Warehouse Operations")
st.caption("Electronics E-commerce Fulfillment Warehouse")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


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
# ANALYTICS + ALERTS
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
    st.write("Ask questions about stock, suppliers, locations, shipments, warehouse documents, and restock priority.")

    with st.expander("Sample Questions"):
        st.write("- Which items are below reorder threshold?")
        st.write("- Which shipments are delayed?")
        st.write("- What is the process for damaged goods?")
        st.write("- Which items should be restocked first?")
        st.write("- Which low stock items are affected by delayed shipments?")

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
        st.success("Chat history cleared.")
        st.rerun()

    if submit_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    final_answer, selected_tools, tool_results, resolved_question = run_langgraph_question(question)

                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

                    st.success("Answer:")
                    st.markdown(final_answer)

                    st.info(f"Resolved question: {resolved_question}")
                    st.info(f"LangGraph selected tools: {', '.join(selected_tools)}")

                    st.subheader("Structured Results")
                    shown_any_table = False

                    for tool_name, tool_result in tool_results.items():
                        display_df = tool_result.get("display_df")
                        if display_df is not None:
                            st.markdown(f"### {tool_name.title()} Results")
                            st.dataframe(display_df, width="stretch")
                            shown_any_table = True

                    if not shown_any_table:
                        st.write("No tabular results available for this query.")

                    with st.expander("See raw tool outputs"):
                        for tool_name, tool_result in tool_results.items():
                            st.markdown(f"### {tool_name.title()} Tool Output")
                            st.text(tool_result.get("text_output", ""))

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
Warehouse operations data is often split across inventory tables, shipment records, and operational documents. Staff must manually search across multiple sources to answer common operational questions and prioritize action.

### Proposed Solution
This project builds a private local AI copilot that:
- reads structured warehouse data
- retrieves warehouse knowledge documents
- uses LangGraph to orchestrate multi-agent routing
- includes decision intelligence for restock prioritization
- returns grounded responses through graph-driven execution

### Key Features
- Inventory admin module
- AI copilot with LangGraph orchestration
- Warehouse knowledge base
- Analytics dashboard
- Smart alerts and recommendations
- Decision support for restock priority
- Private local architecture

### Agentic AI Aspect
The system is agentic because it does not answer blindly. It analyzes the question, routes it through graph nodes such as inventory, shipment, document, and decision intelligence, gathers relevant data, and produces a final grounded response.

### Privacy Aspect
- local database
- local documents
- local LLM-compatible architecture
- no cloud API required for core data

### Tech Stack
- Python
- Streamlit
- SQLite
- LangGraph
- Pandas
""")