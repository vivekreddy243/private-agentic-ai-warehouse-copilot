import sqlite3
from pathlib import Path
import pandas as pd
import re

DB_PATH = "data/warehouse.db"
DOCS_DIR = Path("data/docs")


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


def inventory_tool_logic(question: str):
    df = get_products_df()
    q = question.lower().strip()

    if "low stock" in q or "below reorder threshold" in q or "restock" in q:
        result = df[df["quantity"] < df["reorder_threshold"]][
            ["sku", "name", "quantity", "reorder_threshold", "location", "supplier"]
        ]
        if result.empty:
            return {"display_df": None, "text_output": "No items are currently below reorder threshold."}
        return {"display_df": result, "text_output": result.to_string(index=False)}

    elif "where is" in q or "where are" in q:
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
            return {"display_df": None, "text_output": "No matching product location found."}
        return {"display_df": result, "text_output": result.to_string(index=False)}

    elif "supplier" in q or "who supplies" in q:
        result = df[["sku", "name", "supplier", "location", "quantity"]]
        return {"display_df": result, "text_output": result.to_string(index=False)}

    result = df[["sku", "name", "category", "quantity", "location", "supplier"]]
    return {"display_df": result, "text_output": result.to_string(index=False)}


def shipment_tool_logic(question: str):
    df = get_shipments_df()
    q = question.lower().strip()

    if "delayed" in q or "shipment" in q or "delivery" in q:
        result = df[df["status"].str.lower() == "delayed"][
            ["shipment_id", "supplier", "status", "expected_date", "notes"]
        ]
        if result.empty:
            return {"display_df": None, "text_output": "No delayed shipments found."}
        return {"display_df": result, "text_output": result.to_string(index=False)}

    result = df[["shipment_id", "supplier", "status", "expected_date", "notes"]]
    return {"display_df": result, "text_output": result.to_string(index=False)}


# -----------------------------
# DOCUMENT RELEVANCE HELPERS
# -----------------------------
STOPWORDS = {
    "the", "is", "are", "a", "an", "of", "for", "to", "in", "on", "and", "or",
    "what", "which", "show", "tell", "me", "about", "this", "that", "these",
    "those", "process", "policy", "document", "file", "please", "do", "we"
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def extract_keywords(question: str):
    words = normalize_text(question).split()
    keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return list(dict.fromkeys(keywords))


def split_into_chunks(text: str, max_chars: int = 500):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def score_text(text: str, keywords: list[str]) -> int:
    norm = normalize_text(text)
    score = 0
    for kw in keywords:
        if kw in norm:
            score += norm.count(kw)
    return score


def document_tool_logic(question: str):
    keywords = extract_keywords(question)

    if not DOCS_DIR.exists():
        return {"display_df": None, "text_output": "Document directory not found."}

    scored_chunks = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                chunks = split_into_chunks(content)

                best_chunk = None
                best_score = 0

                for chunk in chunks:
                    score = score_text(chunk, keywords)
                    if score > best_score:
                        best_score = score
                        best_chunk = chunk

                file_bonus = score_text(file_path.name, keywords)
                total_score = best_score + (2 * file_bonus)

                if total_score > 0 and best_chunk:
                    scored_chunks.append({
                        "filename": file_path.name,
                        "score": total_score,
                        "best_chunk": best_chunk
                    })
            except Exception:
                continue

    if not scored_chunks:
        return {"display_df": None, "text_output": "No relevant document content found."}

    scored_chunks = sorted(scored_chunks, key=lambda x: x["score"], reverse=True)

    # Keep only top 2 matches for cleaner output
    top_matches = scored_chunks[:2]

    df = pd.DataFrame([
        {
            "filename": item["filename"],
            "score": item["score"],
            "preview": item["best_chunk"][:220]
        }
        for item in top_matches
    ])

    combined_text = "\n\n".join(
        [
            f"FILE: {item['filename']}\nRELEVANCE SCORE: {item['score']}\nBEST MATCHING CONTENT:\n{item['best_chunk']}"
            for item in top_matches
        ]
    )

    return {
        "display_df": df,
        "text_output": combined_text
    }


def decision_tool_logic():
    df = get_products_df()
    shipments_df = get_shipments_df()
    sales_df = pd.DataFrame([
        {"product": "Wireless Mouse", "month": "Jan", "units_sold": 32},
        {"product": "Wireless Mouse", "month": "Feb", "units_sold": 28},
        {"product": "Wireless Mouse", "month": "Mar", "units_sold": 35},
        {"product": "Mechanical Keyboard", "month": "Jan", "units_sold": 20},
        {"product": "Mechanical Keyboard", "month": "Feb", "units_sold": 18},
        {"product": "Mechanical Keyboard", "month": "Mar", "units_sold": 22},
        {"product": "Portable SSD", "month": "Jan", "units_sold": 12},
        {"product": "Portable SSD", "month": "Feb", "units_sold": 15},
        {"product": "Portable SSD", "month": "Mar", "units_sold": 19},
        {"product": "Thermal Printer", "month": "Jan", "units_sold": 4},
        {"product": "Thermal Printer", "month": "Feb", "units_sold": 3},
        {"product": "Thermal Printer", "month": "Mar", "units_sold": 5},
        {"product": "Barcode Scanner", "month": "Jan", "units_sold": 5},
        {"product": "Barcode Scanner", "month": "Feb", "units_sold": 6},
        {"product": "Barcode Scanner", "month": "Mar", "units_sold": 7},
        {"product": "Laptop Stand", "month": "Jan", "units_sold": 9},
        {"product": "Laptop Stand", "month": "Feb", "units_sold": 11},
        {"product": "Laptop Stand", "month": "Mar", "units_sold": 13},
        {"product": "Noise-Cancelling Headphones", "month": "Jan", "units_sold": 8},
        {"product": "Noise-Cancelling Headphones", "month": "Feb", "units_sold": 10},
        {"product": "Noise-Cancelling Headphones", "month": "Mar", "units_sold": 12},
    ])

    low_stock_df = df[df["quantity"] < df["reorder_threshold"]].copy()

    if low_stock_df.empty:
        return {
            "display_df": None,
            "text_output": "No urgent restock recommendations at the moment."
        }

    sales_summary = sales_df.groupby("product", as_index=False)["units_sold"].sum()
    sales_summary = sales_summary.rename(columns={"product": "name", "units_sold": "sales_score"})

    delayed_suppliers = set(
        shipments_df[shipments_df["status"].str.lower() == "delayed"]["supplier"].tolist()
    )

    low_stock_df["reorder_gap"] = low_stock_df["reorder_threshold"] - low_stock_df["quantity"]
    low_stock_df = low_stock_df.merge(sales_summary, on="name", how="left")
    low_stock_df["sales_score"] = low_stock_df["sales_score"].fillna(0)
    low_stock_df["delay_risk_score"] = low_stock_df["supplier"].apply(
        lambda s: 5 if s in delayed_suppliers else 0
    )

    low_stock_df["priority_score"] = (
        2 * low_stock_df["reorder_gap"]
        + 0.3 * low_stock_df["sales_score"]
        + low_stock_df["delay_risk_score"]
    )

    result = low_stock_df.sort_values(by="priority_score", ascending=False)[
        [
            "sku",
            "name",
            "quantity",
            "reorder_threshold",
            "supplier",
            "reorder_gap",
            "sales_score",
            "delay_risk_score",
            "priority_score",
        ]
    ]

    return {
        "display_df": result,
        "text_output": result.to_string(index=False)
    }