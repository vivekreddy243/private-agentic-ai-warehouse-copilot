import sys
from pathlib import Path
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from app.agents.langgraph_runner import run_langgraph_question
from app.agents.tool_helpers import evaluate_forecast_model
from app.services.dynamic_query_engine import answer_dynamic_question
from app.services.local_router import classify_route, get_out_of_scope_response

DB_PATH = "data/warehouse.db"
DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Warehouse Copilot",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.2rem;
}
.sub-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #f8fafc, #eef2ff);
    padding: 1rem 1.1rem;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
}
.metric-label {
    font-size: 0.85rem;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
}
.answer-box {
    background-color: #f8fafc;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border-left: 6px solid #2563eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}
.user-msg {
    background: #eff6ff;
    padding: 0.85rem 1rem;
    border-radius: 14px;
    margin-bottom: 0.45rem;
    border: 1px solid #bfdbfe;
}
.bot-msg {
    background: #f8fafc;
    padding: 0.85rem 1rem;
    border-radius: 14px;
    margin-bottom: 0.65rem;
    border-left: 4px solid #2563eb;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
}
.small-muted {
    font-size: 0.9rem;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def safe_table_df(table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        exists = cur.fetchone() is not None
        if not exists:
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()

def get_products_df():
    return safe_table_df("products")

def get_shipments_df():
    return safe_table_df("shipments")

def get_sales_df():
    return safe_table_df("sales")

def get_outflow_df():
    return safe_table_df("outflow")

def get_inflow_df():
    return safe_table_df("inflow")

def build_dashboard_metrics():
    products_df = get_products_df()
    shipments_df = get_shipments_df()
    sales_df = get_sales_df()
    outflow_df = get_outflow_df()
    inflow_df = get_inflow_df()

    total_products = len(products_df) if not products_df.empty else 0

    low_stock_df = pd.DataFrame()
    low_stock_count = 0
    total_suppliers = 0
    category_summary = pd.DataFrame()
    supplier_summary = pd.DataFrame()

    if not products_df.empty:
        low_stock_df = products_df[products_df["quantity"] < products_df["reorder_threshold"]].copy()
        low_stock_count = len(low_stock_df)
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

    delayed_shipments_df = pd.DataFrame()
    delayed_shipments_count = 0
    if not shipments_df.empty:
        delayed_shipments_df = shipments_df[
            shipments_df["status"].astype(str).str.lower() == "delayed"
        ].copy()
        delayed_shipments_count = len(delayed_shipments_df)

    monthly_sales = pd.DataFrame(columns=["month", "quantity_sold"])
    product_sales = pd.DataFrame(columns=["name", "quantity_sold"])

    if not sales_df.empty:
        sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"], errors="coerce")
        sales_df = sales_df.dropna(subset=["sale_date"]).copy()
        sales_df["month"] = sales_df["sale_date"].dt.strftime("%b")

        monthly_sales = (
            sales_df.groupby("month", as_index=False)["quantity_sold"]
            .sum()
        )

        product_sales = (
            sales_df.groupby("name", as_index=False)["quantity_sold"]
            .sum()
            .sort_values(by="quantity_sold", ascending=False)
        )

    monthly_outflow = pd.DataFrame(columns=["month", "quantity_out"])
    if not outflow_df.empty:
        outflow_df["outflow_date"] = pd.to_datetime(outflow_df["outflow_date"], errors="coerce")
        outflow_df = outflow_df.dropna(subset=["outflow_date"]).copy()
        outflow_df["month"] = outflow_df["outflow_date"].dt.strftime("%b")

        monthly_outflow = (
            outflow_df.groupby("month", as_index=False)["quantity_out"]
            .sum()
        )

    monthly_inflow = pd.DataFrame(columns=["month", "quantity_in"])
    if not inflow_df.empty:
        inflow_df["inflow_date"] = pd.to_datetime(inflow_df["inflow_date"], errors="coerce")
        inflow_df = inflow_df.dropna(subset=["inflow_date"]).copy()
        inflow_df["month"] = inflow_df["inflow_date"].dt.strftime("%b")

        monthly_inflow = (
            inflow_df.groupby("month", as_index=False)["quantity_in"]
            .sum()
        )

    return {
        "products_df": products_df,
        "shipments_df": shipments_df,
        "sales_df": sales_df,
        "outflow_df": outflow_df,
        "inflow_df": inflow_df,
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
        "monthly_outflow": monthly_outflow,
        "monthly_inflow": monthly_inflow,
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
        delayed_suppliers = ", ".join(delayed_shipments_df["supplier"].dropna().unique().tolist())
        recommendations.append(
            f"Follow up with delayed shipment suppliers: {delayed_suppliers}."
        )

    if not low_stock_df.empty and not delayed_shipments_df.empty:
        recommendations.append(
            "Review whether delayed shipments are affecting low-stock products."
        )

    if not dashboard["product_sales"].empty:
        top_sales = dashboard["product_sales"].head(3)
        recommendations.append(
            "Top-selling products to monitor closely: "
            + ", ".join(top_sales["name"].tolist()) + "."
        )

    if not alerts:
        alerts.append("No major operational alerts right now.")

    if not recommendations:
        recommendations.append("No urgent recommendations at the moment.")

    return alerts, recommendations

def get_system_status():
    db_ok = Path(DB_PATH).exists()
    docs_count = len([f for f in DOCS_DIR.iterdir() if f.is_file()]) if DOCS_DIR.exists() else 0

    try:
        ml_eval = evaluate_forecast_model()
        metrics = ml_eval.get("metrics", {})
    except Exception:
        metrics = {}

    return {
        "db_status": "Connected" if db_ok else "Missing",
        "docs_count": docs_count,
        "routing_accuracy": "90.0%",
        "ml_mae": metrics.get("MAE", "N/A"),
        "ml_mse": metrics.get("MSE", "N/A"),
        "ml_r2": metrics.get("R2", "N/A"),
    }

dashboard = build_dashboard_metrics()

st.markdown('<div class="main-title">Agentic AI Warehouse Copilot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Intelligent warehouse operations assistant for inventory visibility, shipment monitoring, sales/outflow tracking, SOP retrieval, and demand-aware restock decisions.</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Total Products</div><div class="metric-value">{dashboard["total_products"]}</div></div>',
        unsafe_allow_html=True
    )
with m2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Low Stock Items</div><div class="metric-value">{dashboard["low_stock_count"]}</div></div>',
        unsafe_allow_html=True
    )
with m3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Delayed Shipments</div><div class="metric-value">{dashboard["delayed_shipments_count"]}</div></div>',
        unsafe_allow_html=True
    )
with m4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Suppliers</div><div class="metric-value">{dashboard["total_suppliers"]}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📦 Inventory",
        "🤖 AI Copilot",
        "📚 Knowledge Base",
        "📊 Analytics",
        "🚨 Alerts",
        "🧪 Evaluation",
        "📝 Project Info",
    ]
)

with tab1:
    st.subheader("Inventory Management")
    st.write("View current inventory and manage warehouse products.")

    left, right = st.columns([1.8, 1])

    with left:
        st.markdown("### Current Inventory")
        df = get_products_df()
        st.dataframe(df, width="stretch", height=420)

    with right:
        st.markdown("### Add New Product")
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
    st.write("Ask warehouse questions across inventory, shipments, sales, inflow, outflow, documents, and restock intelligence.")

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("### Suggested Questions")
        st.markdown("- What products are there in the warehouse?")
        st.markdown("- What items should I restock immediately?")
        st.markdown("- Which shipments are delayed?")
        st.markdown("- What are the sales this month?")
    with s2:
        st.markdown("### More Examples")
        st.markdown("- What is the outflow this month?")
        st.markdown("- What is the inflow this month?")
        st.markdown("- Which supplier affects low stock the most?")
        st.markdown("- What is SKU?")

    if st.session_state.chat_history:
        st.markdown("### Recent Conversation")
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                st.markdown(
                    f"<div class='user-msg'><b>You:</b> {msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='bot-msg'><b>Copilot:</b><br>{msg['content']}</div>",
                    unsafe_allow_html=True
                )

    question = st.text_input(
        "Ask a warehouse question",
        placeholder="Example: What are the sales this month?"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        submit_clicked = st.button("Submit Question", width="stretch")

    with col2:
        clear_clicked = st.button("Clear Chat History", width="stretch")

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
                    resolved_question = question
                    route_info = classify_route(question)
                    route = route_info["route"]

                    if route == "out_of_scope":
                        final_answer = get_out_of_scope_response()

                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)
                        with st.expander("See reasoning details"):
                            st.write(f"Resolved question: {resolved_question}")
                            st.write(f"Route selected: {route}")
                            st.write(f"Route confidence: {route_info['confidence']}")
                            if route_info.get("matched_example"):
                                st.write(f"Closest route example: {route_info['matched_example']}")

                        st.markdown("### Structured Results")
                        st.info("No tabular results available for this query.")

                    elif route == "document":
                        final_answer, selected_tools, tool_results, resolved_question = run_langgraph_question(question)

                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

                        with st.expander("See reasoning details"):
                            st.write(f"Resolved question: {resolved_question}")
                            st.write(f"Route selected: {route}")
                            st.write(f"Route confidence: {route_info['confidence']}")
                            if route_info.get("matched_example"):
                                st.write(f"Closest route example: {route_info['matched_example']}")
                            st.write(f"LangGraph tools used: {', '.join(selected_tools)}")

                        st.markdown("### Structured Results")
                        shown_any_table = False
                        for tool_name, tool_result in tool_results.items():
                            display_df = tool_result.get("display_df")
                            if display_df is not None and not display_df.empty:
                                st.markdown(f"#### {tool_name.title()} Results")
                                st.dataframe(display_df, width="stretch")
                                shown_any_table = True

                        if not shown_any_table:
                            st.info("No tabular results available for this query.")

                    else:
                        final_answer, direct_tables, structured_meta = answer_dynamic_question(question)

                        st.session_state.chat_history.append({"role": "user", "content": question})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_answer})

                        st.markdown("### Answer")
                        st.markdown(f'<div class="answer-box">{final_answer}</div>', unsafe_allow_html=True)

                        with st.expander("See reasoning details"):
                            st.write(f"Resolved question: {resolved_question}")
                            st.write(f"Route selected: {route}")
                            st.write(f"Route confidence: {route_info['confidence']}")
                            if route_info.get("matched_example"):
                                st.write(f"Closest route example: {route_info['matched_example']}")
                            st.write(f"Detected structured intent: {structured_meta['intent']}")
                            st.write(f"Extracted entities: {structured_meta['entities']}")
                            st.write(f"Intent confidence: {structured_meta['confidence']}")
                            st.write(f"SQL template used: {structured_meta['sql_template']}")

                        st.markdown("### Structured Results")
                        shown_any_table = False
                        for title, display_df in direct_tables.items():
                            if display_df is not None and not display_df.empty:
                                st.markdown(f"#### {title}")
                                st.dataframe(display_df, width="stretch")
                                shown_any_table = True

                        if not shown_any_table:
                            st.info("No tabular results available for this query.")

                except Exception as e:
                    st.error(f"Error: {e}")

with tab3:
    st.subheader("Warehouse Knowledge Base")
    st.write("Upload local warehouse SOPs, supplier notes, policy documents, or structured inventory/sales/inflow/outflow CSV files.")

    upload_type = st.selectbox(
        "Select upload type",
        ["Document (.txt)", "Inventory CSV", "Sales CSV", "Inflow CSV", "Outflow CSV"]
    )

    if upload_type == "Document (.txt)":
        uploaded_file = st.file_uploader("Upload a text document", type=["txt"], key="txt_upload")

        if uploaded_file is not None:
            save_path = DOCS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Document uploaded: {uploaded_file.name}")

    elif upload_type == "Inventory CSV":
        uploaded_file = st.file_uploader("Upload inventory CSV", type=["csv"], key="inventory_csv_upload")

        if uploaded_file is not None:
            try:
                inventory_df = pd.read_csv(uploaded_file)

                required_cols = {"sku", "name", "category", "quantity", "reorder_threshold", "location", "supplier"}
                if not required_cols.issubset(set(inventory_df.columns)):
                    st.error(f"Inventory CSV must contain columns: {sorted(required_cols)}")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    inventory_df.to_sql("products", conn, if_exists="append", index=False)
                    conn.close()
                    st.success(f"Inventory data uploaded successfully. Inserted {len(inventory_df)} rows.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to upload inventory CSV: {e}")

    elif upload_type == "Sales CSV":
        uploaded_file = st.file_uploader("Upload sales CSV", type=["csv"], key="sales_csv_upload")

        if uploaded_file is not None:
            try:
                sales_df = pd.read_csv(uploaded_file)

                required_cols = {"sku", "name", "quantity_sold", "revenue", "sale_date"}
                if not required_cols.issubset(set(sales_df.columns)):
                    st.error(f"Sales CSV must contain columns: {sorted(required_cols)}")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    sales_df.to_sql("sales", conn, if_exists="append", index=False)
                    conn.close()
                    st.success(f"Sales data uploaded successfully. Inserted {len(sales_df)} rows.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to upload sales CSV: {e}")

    elif upload_type == "Inflow CSV":
        uploaded_file = st.file_uploader("Upload inflow CSV", type=["csv"], key="inflow_csv_upload")

        if uploaded_file is not None:
            try:
                inflow_df = pd.read_csv(uploaded_file)

                required_cols = {"sku", "name", "quantity_in", "source", "inflow_date"}
                if not required_cols.issubset(set(inflow_df.columns)):
                    st.error(f"Inflow CSV must contain columns: {sorted(required_cols)}")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    inflow_df.to_sql("inflow", conn, if_exists="append", index=False)
                    conn.close()
                    st.success(f"Inflow data uploaded successfully. Inserted {len(inflow_df)} rows.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to upload inflow CSV: {e}")

    elif upload_type == "Outflow CSV":
        uploaded_file = st.file_uploader("Upload outflow CSV", type=["csv"], key="outflow_csv_upload")

        if uploaded_file is not None:
            try:
                outflow_df = pd.read_csv(uploaded_file)

                required_cols = {"sku", "name", "quantity_out", "destination", "outflow_date"}
                if not required_cols.issubset(set(outflow_df.columns)):
                    st.error(f"Outflow CSV must contain columns: {sorted(required_cols)}")
                else:
                    conn = sqlite3.connect(DB_PATH)
                    outflow_df.to_sql("outflow", conn, if_exists="append", index=False)
                    conn.close()
                    st.success(f"Outflow data uploaded successfully. Inserted {len(outflow_df)} rows.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to upload outflow CSV: {e}")

    st.markdown("### Available Documents")
    files = [f.name for f in DOCS_DIR.iterdir() if f.is_file()]
    if files:
        docs_df = pd.DataFrame({"Document Name": files})
        st.dataframe(docs_df, width="stretch", height=180)
    else:
        st.info("No documents uploaded yet.")

    st.markdown("### Structured Data Summary")
    conn = sqlite3.connect(DB_PATH)
    try:
        product_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM products", conn)["cnt"][0]
        sales_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM sales", conn)["cnt"][0]
        inflow_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM inflow", conn)["cnt"][0]
        outflow_count = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM outflow", conn)["cnt"][0]
    finally:
        conn.close()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", product_count)
    c2.metric("Sales Rows", sales_count)
    c3.metric("Inflow Rows", inflow_count)
    c4.metric("Outflow Rows", outflow_count)

with tab4:
    st.subheader("Warehouse Analytics")
    st.write("Operational overview of stock, shipments, sales, inflow, and outflow.")

    dashboard = build_dashboard_metrics()

    row1c1, row1c2, row1c3, row1c4 = st.columns(4)
    row1c1.metric("Total Products", dashboard["total_products"])
    row1c2.metric("Low Stock Items", dashboard["low_stock_count"])
    row1c3.metric("Delayed Shipments", dashboard["delayed_shipments_count"])
    row1c4.metric("Total Suppliers", dashboard["total_suppliers"])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Monthly Sales Trend")
        if dashboard["monthly_sales"].empty:
            st.info("Sales data not available.")
        else:
            st.line_chart(dashboard["monthly_sales"].set_index("month"))

    with c2:
        st.subheader("Product Sales Summary")
        if dashboard["product_sales"].empty:
            st.info("Sales data not available.")
        else:
            st.bar_chart(dashboard["product_sales"].set_index("name"))

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Monthly Inflow")
        if dashboard["monthly_inflow"].empty:
            st.info("Inflow data not available.")
        else:
            st.bar_chart(dashboard["monthly_inflow"].set_index("month"))

    with c4:
        st.subheader("Monthly Outflow")
        if dashboard["monthly_outflow"].empty:
            st.info("Outflow data not available.")
        else:
            st.bar_chart(dashboard["monthly_outflow"].set_index("month"))

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Category-wise Stock Summary")
        if dashboard["category_summary"].empty:
            st.info("Category summary not available.")
        else:
            st.dataframe(dashboard["category_summary"], width="stretch")

    with c6:
        st.subheader("Supplier-wise Quantity Summary")
        if dashboard["supplier_summary"].empty:
            st.info("Supplier summary not available.")
        else:
            st.dataframe(dashboard["supplier_summary"], width="stretch")

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Low Stock Products")
        if dashboard["low_stock_df"].empty:
            st.success("No low stock items.")
        else:
            st.dataframe(
                dashboard["low_stock_df"][["sku", "name", "quantity", "reorder_threshold", "location", "supplier"]],
                width="stretch"
            )

    with c8:
        st.subheader("Delayed Shipments")
        if dashboard["delayed_shipments_df"].empty:
            st.success("No delayed shipments.")
        else:
            available_cols = [c for c in ["shipment_id", "supplier", "status", "expected_date", "notes"] if c in dashboard["delayed_shipments_df"].columns]
            st.dataframe(
                dashboard["delayed_shipments_df"][available_cols],
                width="stretch"
            )

with tab5:
    st.subheader("Smart Alerts & Recommendations")
    st.write("Operational signals derived from inventory, shipments, and sales patterns.")

    dashboard = build_dashboard_metrics()
    alerts, recommendations = build_alerts_and_recommendations(dashboard)

    left, right = st.columns(2)
    with left:
        st.markdown("### Alerts")
        for alert in alerts:
            st.warning(alert)

    with right:
        st.markdown("### Recommendations")
        for rec in recommendations:
            st.success(rec)

with tab6:
    st.subheader("System Evaluation")
    st.write("Routing and ML evaluation summary for the warehouse copilot.")

    status = get_system_status()

    c1, c2, c3 = st.columns(3)
    c1.metric("Routing Accuracy", status["routing_accuracy"])
    c2.metric("ML MAE", status["ml_mae"])
    c3.metric("ML R²", status["ml_r2"])

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("### System Status")
        st.info(f"Database Status: {status['db_status']}")
        st.info(f"Documents Loaded: {status['docs_count']}")

    with right:
        st.markdown("### Evaluation Notes")
        st.markdown("- Routing benchmark size: **30 warehouse questions**")
        st.markdown("- Correct tool-routing matches: **27**")
        st.markdown("- Forecast metrics computed using hold-out monthly demand comparison")
        st.markdown("- Model used: **Linear Regression**")

    try:
        ml_eval = evaluate_forecast_model()
        details_df = ml_eval.get("details_df")
        if details_df is not None and not details_df.empty:
            st.markdown("### Forecast Evaluation Details")
            st.dataframe(details_df, width="stretch")
    except Exception:
        st.info("Forecast evaluation details are not available.")

with tab7:
    st.subheader("Project Overview")
    st.write("Detailed summary of the system design, purpose, and technology stack.")

    st.markdown("""
### Project Title
**Agentic AI-Based Warehouse Copilot for Intelligent Inventory Management and Demand-Aware Restock Decision Support**

### Problem Statement
Warehouse operations data is often split across inventory tables, shipment records, operational documents, sales logs, and stock movement records. Staff must manually search across multiple sources to answer common operational questions and prioritize action.

### Proposed Solution
This project builds a private local AI copilot that:
- reads structured warehouse data
- tracks sales, inflow, and outflow
- retrieves warehouse knowledge documents
- uses LangGraph to orchestrate multi-tool routing
- includes decision intelligence for restock prioritization
- combines rule-based warehouse logic with ML-based demand forecasting

### Key Features
- Inventory management module
- AI copilot with dynamic intent detection and SQL-backed responses
- Warehouse knowledge base
- Analytics dashboard
- Smart alerts and recommendations
- Sales / inflow / outflow visibility
- ML-based restock risk analysis
- Private local architecture

### Agentic AI Aspect
The system is agentic because it does not answer blindly. It analyzes the question, routes it through inventory, shipment, sales, outflow, document, and decision pathways, gathers relevant context, and produces a grounded response.

### Machine Learning Aspect
The project uses a lightweight Linear Regression model to predict next-month demand from historical sales data. This forecast is combined with stock gap and shipment-delay risk to create a restock risk score.

### Privacy Aspect
- local database
- local documents
- local execution
- no cloud API required for core data

### Tech Stack
- Python
- Streamlit
- SQLite
- LangGraph
- Pandas
- NumPy
- scikit-learn
""")
