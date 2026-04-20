import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DB_PATH = "data/warehouse.db"
DOCS_DIR = Path("data/docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


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
    conn = sqlite3.connect(DB_PATH)
    try:
        sales_df = pd.read_sql_query(
            "SELECT name, quantity_sold, sale_date FROM sales",
            conn
        )
    except Exception:
        return pd.DataFrame(
            columns=["product", "month", "month_start", "month_num", "units_sold"]
        )
    finally:
        conn.close()

    if sales_df.empty:
        return pd.DataFrame(
            columns=["product", "month", "month_start", "month_num", "units_sold"]
        )

    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"], errors="coerce")
    sales_df = sales_df.dropna(subset=["sale_date"]).copy()

    if sales_df.empty:
        return pd.DataFrame(
            columns=["product", "month", "month_start", "month_num", "units_sold"]
        )

    sales_df["month_start"] = sales_df["sale_date"].dt.to_period("M").dt.to_timestamp()

    monthly_sales = (
        sales_df.groupby(["name", "month_start"], as_index=False)["quantity_sold"]
        .sum()
        .sort_values(by=["name", "month_start"])
    )

    month_index = {
        month: idx
        for idx, month in enumerate(sorted(monthly_sales["month_start"].unique()), start=1)
    }

    monthly_sales["month"] = monthly_sales["month_start"].dt.strftime("%Y-%m")
    monthly_sales["month_num"] = monthly_sales["month_start"].map(month_index)

    monthly_sales = monthly_sales.rename(
        columns={"name": "product", "quantity_sold": "units_sold"}
    )

    return monthly_sales[["product", "month", "month_start", "month_num", "units_sold"]]


def inventory_tool_logic(question: str):
    df = get_products_df()
    q = question.lower()

    if any(word in q for word in ["restock", "threshold", "low stock", "immediately", "urgent"]):
        result_df = df[df["quantity"] < df["reorder_threshold"]].copy()
        result_df = result_df.sort_values(by="quantity", ascending=True)

        return {
            "text_output": result_df.to_string(index=False) if not result_df.empty else "No low stock items found.",
            "display_df": result_df
        }

    if "supplier" in q:
        result_df = df[["sku", "name", "supplier", "quantity", "location"]].copy()
        return {
            "text_output": result_df.to_string(index=False),
            "display_df": result_df
        }

    if "location" in q or "where" in q or "stored" in q:
        result_df = df[["sku", "name", "location", "quantity"]].copy()
        return {
            "text_output": result_df.to_string(index=False),
            "display_df": result_df
        }

    return {
        "text_output": df.to_string(index=False),
        "display_df": df
    }


def shipment_tool_logic(question: str):
    df = get_shipments_df()
    q = question.lower()

    if any(word in q for word in ["delayed", "shipment", "shipments", "delivery", "expected date"]):
        result_df = df[df["status"].str.lower() == "delayed"].copy()
        return {
            "text_output": result_df.to_string(index=False) if not result_df.empty else "No delayed shipments found.",
            "display_df": result_df
        }

    return {
        "text_output": df.to_string(index=False),
        "display_df": df
    }


def document_tool_logic(question: str):
    query_terms = question.lower().split()
    matches = []

    for file_path in DOCS_DIR.glob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        content_lower = content.lower()
        score = sum(1 for term in query_terms if term in content_lower)

        if score > 0:
            matches.append({
                "file": file_path.name,
                "score": score,
                "content": content.strip()
            })

    matches = sorted(matches, key=lambda x: x["score"], reverse=True)

    if not matches:
        return {
            "text_output": "No relevant warehouse document content found.",
            "display_df": pd.DataFrame(columns=["file", "score", "content"])
        }

    top_matches = matches[:2]

    lines = []
    display_rows = []

    for match in top_matches:
        lines.append(f"FILE: {match['file']}")
        lines.append(f"RELEVANCE SCORE: {match['score']}")
        lines.append(f"BEST MATCHING CONTENT: {match['content']}")
        lines.append("")

        display_rows.append({
            "file": match["file"],
            "score": match["score"],
            "content": match["content"]
        })

    return {
        "text_output": "\n".join(lines).strip(),
        "display_df": pd.DataFrame(display_rows)
    }


def forecast_next_month_demand(sales_df: pd.DataFrame) -> pd.DataFrame:
    if sales_df.empty:
        return pd.DataFrame(
            columns=["product", "predicted_next_month_demand", "trend_slope"]
        )

    df = sales_df.copy()

    forecast_rows = []

    for product, group in df.groupby("product"):
        group = group.sort_values("month_num")
        X = group[["month_num"]].values
        y = group["units_sold"].values

        if len(group) >= 2:
            model = LinearRegression()
            model.fit(X, y)
            next_month_num = float(group["month_num"].max() + 1)
            predicted = float(model.predict(np.array([[next_month_num]]))[0])
            trend_slope = float(model.coef_[0])
        else:
            predicted = float(group["units_sold"].iloc[-1])
            trend_slope = 0.0

        forecast_rows.append({
            "product": product,
            "predicted_next_month_demand": round(max(predicted, 0), 2),
            "trend_slope": round(trend_slope, 2),
        })

    return pd.DataFrame(forecast_rows)


def build_restock_risk_scores():
    products_df = get_products_df()
    shipments_df = get_shipments_df()
    sales_df = get_sales_df()
    forecast_df = forecast_next_month_demand(sales_df)

    delayed_suppliers = set(
        shipments_df.loc[
            shipments_df["status"].str.lower() == "delayed", "supplier"
        ].dropna().tolist()
    )

    merged = products_df.merge(
        forecast_df,
        left_on="name",
        right_on="product",
        how="left"
    )

    merged["stock_gap"] = (merged["reorder_threshold"] - merged["quantity"]).clip(lower=0)
    merged["shipment_delay_flag"] = merged["supplier"].apply(
        lambda s: 1 if s in delayed_suppliers else 0
    )

    for col in ["stock_gap", "predicted_next_month_demand"]:
        max_val = merged[col].max()
        if pd.notna(max_val) and max_val > 0:
            merged[f"{col}_norm"] = merged[col] / max_val
        else:
            merged[f"{col}_norm"] = 0.0

    merged["risk_score"] = (
        0.45 * merged["stock_gap_norm"] +
        0.35 * merged["predicted_next_month_demand_norm"] +
        0.20 * merged["shipment_delay_flag"]
    )

    merged["risk_score"] = merged["risk_score"].round(3)

    result = merged[
        [
            "sku", "name", "quantity", "reorder_threshold", "supplier",
            "predicted_next_month_demand", "trend_slope",
            "shipment_delay_flag", "risk_score"
        ]
    ].sort_values(by="risk_score", ascending=False)

    return result


def decision_tool_logic():
    risk_df = build_restock_risk_scores()
    high_priority_df = risk_df[risk_df["risk_score"] > 0].copy()

    if high_priority_df.empty:
        return {
            "text_output": "No urgent restock risks were detected.",
            "display_df": risk_df
        }

    top_items = high_priority_df.head(5)

    immediate_df = high_priority_df[
        high_priority_df["quantity"] < high_priority_df["reorder_threshold"]
    ].head(4)

    future_risk_df = high_priority_df[
        high_priority_df["quantity"] >= high_priority_df["reorder_threshold"]
    ].head(3)

    lines = []

    if not immediate_df.empty:
        lines.append("Immediate Action Required:")
        for _, row in immediate_df.iterrows():
            lines.append(
                f"- {row['name']} → low stock ({row['quantity']} left, threshold {row['reorder_threshold']})"
            )

    if not future_risk_df.empty:
        lines.append("")
        lines.append("Upcoming Risk to Watch:")
        for _, row in future_risk_df.iterrows():
            reasons = []

            if row["trend_slope"] > 0:
                reasons.append("demand is increasing")

            if row["shipment_delay_flag"] == 1:
                reasons.append("supplier delay risk")

            if not reasons:
                reasons.append("watch inventory trend")

            lines.append(
                f"- {row['name']} → " + ", ".join(reasons)
            )

    lines.append("")
    lines.append("Recommendation: restock low-stock items now and monitor the high-demand products to avoid future shortages.")

    return {
        "text_output": "\n".join(lines),
        "display_df": high_priority_df
    }

def evaluate_forecast_model():
    sales_df = get_sales_df().copy()

    if sales_df.empty or sales_df["month_num"].nunique() < 3:
        return {
            "metrics": {
                "MAE": None,
                "MSE": None,
                "R2": None
            },
            "details_df": pd.DataFrame(
                columns=["product", "predicted_latest_demand", "actual_latest_demand", "absolute_error"]
            )
        }

    latest_month_num = sales_df["month_num"].max()
    latest_month_label = sales_df.loc[
        sales_df["month_num"] == latest_month_num, "month"
    ].iloc[0]

    train_df = sales_df[sales_df["month_num"] < latest_month_num].copy()
    test_df = sales_df[sales_df["month_num"] == latest_month_num].copy()

    predictions = []
    actuals = []
    eval_rows = []

    for product in train_df["product"].unique():
        train_group = train_df[train_df["product"] == product].sort_values("month_num")
        test_group = test_df[test_df["product"] == product].sort_values("month_num")

        if len(train_group) >= 2 and not test_group.empty:
            X_train = train_group[["month_num"]].values
            y_train = train_group["units_sold"].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = float(model.predict(np.array([[latest_month_num]]))[0])
            actual = float(test_group["units_sold"].iloc[0])

            predictions.append(pred)
            actuals.append(actual)

            eval_rows.append({
                "product": product,
                "predicted_latest_demand": round(pred, 2),
                "actual_latest_demand": round(actual, 2),
                "absolute_error": round(abs(actual - pred), 2),
            })

    if not predictions:
        return {
            "metrics": {
                "MAE": None,
                "MSE": None,
                "R2": None
            },
            "details_df": pd.DataFrame(
                columns=["product", "predicted_latest_demand", "actual_latest_demand", "absolute_error"]
            )
        }

    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

    return {
        "metrics": {
            "MAE": round(float(mae), 3),
            "MSE": round(float(mse), 3),
            "R2": round(float(r2), 3),
        },
        "details_df": pd.DataFrame(eval_rows),
        "evaluated_month": latest_month_label
    }
