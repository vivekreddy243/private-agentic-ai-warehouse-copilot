import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

from app.services.local_semantic_matcher import match_examples, normalize_text
from app.services.query_templates import build_query

DB_PATH = "data/warehouse.db"

MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

WAREHOUSE_TERMS = {
    "sku": "SKU stands for Stock Keeping Unit. It is a unique code used to identify and track a product in the warehouse.",
    "reorder threshold": "Reorder threshold is the minimum stock level at which replenishment should be triggered.",
    "outflow": "Outflow means items moving out of warehouse inventory for dispatch, transfer, or delivery.",
    "inflow": "Inflow means incoming stock added to warehouse inventory.",
    "inventory": "Inventory refers to the products and materials currently stored in the warehouse.",
    "shipment delay": "Shipment delay means a shipment has not arrived by its expected date and may affect warehouse operations.",
    "restocking": "Restocking means replenishing product inventory when stock levels fall near or below the reorder threshold.",
}

UNSUPPORTED_WAREHOUSE_RESPONSE = "I couldn’t confidently map that warehouse question to a supported structured query in the current system."
NO_MATCHING_PRODUCTS_RESPONSE = "I found no matching products for that item type in the current warehouse inventory."
INTENT_CONFIDENCE_THRESHOLD = 0.57

COUNT_PHRASES = ("how many", "number of", "total count of", "total number of")
GENERIC_PRODUCT_TERMS = {
    "product", "products", "item", "items", "goods", "inventory", "stock", "warehouse",
}
CATEGORY_TERMS = {
    "electronics", "electronic", "accessories", "accessory", "warehouse supplies",
    "warehouse supply", "networking", "office devices", "office device",
}

INTENT_EXAMPLES = {
    "supplier_count": [
        "how many suppliers are there",
        "how many different suppliers do we have",
        "total suppliers",
    ],
    "supplier_list": [
        "list suppliers",
        "show warehouse suppliers",
        "which vendors do we have",
    ],
    "product_count": [
        "how many products do we have",
        "total products",
        "number of products",
    ],
    "product_list": [
        "what products are there in the warehouse",
        "list products",
        "what items are there",
        "show warehouse products",
    ],
    "product_type_summary": [
        "what type of products are there",
        "what categories of products are there",
        "what kinds of products are in the warehouse",
    ],
    "filtered_product_count": [
        "how many keyboards are there in warehouse",
        "how many scanners do we have",
        "how many printers are there",
        "how many monitors are in stock",
        "how many electronics items do we have",
    ],
    "filtered_product_list": [
        "show all keyboards",
        "list all scanners",
        "show electronics products",
        "show all warehouse supplies items",
        "list products from TechSupply",
        "show products in Rack-A1",
    ],
    "sales_total": [
        "what are the sales this month",
        "total sales this month",
        "revenue this month",
        "units sold this month",
    ],
    "sales_top_item": [
        "what is the most selling item in the warehouse",
        "top selling item",
        "which product sold the most",
        "highest selling product",
    ],
    "sales_by_month_summary": [
        "which month has highest sales",
        "compare sales between march and april",
        "show monthly sales trend",
    ],
    "inflow_total": [
        "what is the inflow this month",
        "total inflow this month",
        "stock received this month",
    ],
    "outflow_total": [
        "what is the outflow this month",
        "total outflow this month",
        "stock moved out this month",
    ],
    "low_stock": [
        "which items are low in stock",
        "products below reorder threshold",
        "critical low-stock items",
    ],
    "restock_priority": [
        "what should i restock immediately",
        "which items should i restock immediately",
        "restock priority list",
    ],
    "delayed_shipments": [
        "which shipments are delayed",
        "delayed shipment list",
        "shipment issues",
    ],
    "shipment_status_lookup": [
        "what is the status of shipment ship-2001",
        "show shipment details for ship-2045",
        "look up shipment ship-2001",
    ],
    "supplier_risk": [
        "which supplier affects low stock the most",
        "supplier causing low stock issues",
    ],
    "location_count": [
        "how many locations are there",
        "how many different storage locations",
    ],
    "rack_type_count": [
        "how many rack types are there",
        "number of rack types",
    ],
    "location_lookup": [
        "where is wireless mouse stored",
        "where is barcode scanner stored",
        "which rack is thermal printer in",
    ],
    "location_usage_summary": [
        "what products are in rack-a1",
        "which locations contain low-stock items",
        "are there any empty locations",
        "show location usage summary",
    ],
    "category_summary": [
        "show stock by category",
        "category-wise inventory summary",
        "which category has highest quantity",
    ],
    "stock_level_lookup": [
        "what is the current stock of mechanical keyboard",
        "how much stock is left for wireless mouse",
        "stock level for barcode scanner",
    ],
    "zero_stock": [
        "which items are out of stock",
        "show zero stock items",
        "what products have zero quantity",
    ],
    "unsold_items": [
        "which items were not sold this month",
        "unsold products in april",
    ],
    "warehouse_concept": [
        "what is sku",
        "what is reorder threshold",
        "what is inflow",
        "what is outflow",
        "what is inventory",
        "what is shipment delay",
        "what is restocking",
    ],
}

INTENT_CUE_WORDS = {
    "supplier_count": {"supplier", "count"},
    "supplier_list": {"supplier", "list"},
    "product_count": {"product", "count"},
    "product_list": {"product", "list"},
    "product_type_summary": {"product", "category"},
    "filtered_product_count": {"count"},
    "filtered_product_list": {"product", "list"},
    "sales_total": {"sales"},
    "sales_top_item": {"sales", "top"},
    "sales_by_month_summary": {"sales", "compare"},
    "inflow_total": {"inflow"},
    "outflow_total": {"outflow"},
    "low_stock": {"stock", "restock"},
    "restock_priority": {"restock", "priority"},
    "delayed_shipments": {"shipment", "delayed"},
    "shipment_status_lookup": {"shipment", "status"},
    "supplier_risk": {"supplier", "restock"},
    "location_count": {"location", "count"},
    "rack_type_count": {"rack", "count"},
    "location_lookup": {"location", "where"},
    "location_usage_summary": {"location", "summary"},
    "category_summary": {"category"},
    "stock_level_lookup": {"stock"},
    "zero_stock": {"stock", "zero"},
    "unsold_items": {"sales", "product"},
    "warehouse_concept": {"sku", "inventory", "inflow", "outflow", "threshold", "restocking"},
}


def run_sql(query: str, params=()) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def get_unsupported_warehouse_response() -> str:
    return UNSUPPORTED_WAREHOUSE_RESPONSE


def _singularize_phrase(phrase: str) -> str:
    phrase = phrase.strip()
    if phrase in CATEGORY_TERMS:
        return phrase
    words = []
    for word in phrase.split():
        if word in CATEGORY_TERMS:
            words.append(word)
        elif len(word) > 4 and word.endswith("ies"):
            words.append(word[:-3] + "y")
        elif len(word) > 3 and word.endswith("es") and not word.endswith("ses"):
            words.append(word[:-2])
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            words.append(word[:-1])
        else:
            words.append(word)
    return " ".join(words).strip()


def _latest_month_from_table(table: str, date_column: str) -> Optional[str]:
    try:
        df = run_sql(f"SELECT {date_column} FROM {table}")
    except Exception:
        return None

    if df.empty:
        return None

    parsed = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if parsed.empty:
        return None

    return str(parsed.dt.to_period("M").max())


def _normalize_months(question: str) -> list[str]:
    q = normalize_text(question)
    values = []
    for word, mm in MONTH_MAP.items():
        if word in q.split():
            month_value = f"{datetime.now().year}-{mm}"
            if month_value not in values:
                values.append(month_value)
    return values


def extract_time_entities(question: str, intent: str) -> dict:
    q = normalize_text(question)
    explicit_months = _normalize_months(question)
    result = {"time_filter": None, "comparison_months": []}

    if len(explicit_months) >= 2:
        result["comparison_months"] = explicit_months[:2]
        result["time_filter"] = explicit_months[0]
        return result

    if explicit_months:
        result["time_filter"] = explicit_months[0]
        return result

    table_map = {
        "sales_total": ("sales", "sale_date"),
        "sales_top_item": ("sales", "sale_date"),
        "sales_by_month_summary": ("sales", "sale_date"),
        "inflow_total": ("inflow", "inflow_date"),
        "outflow_total": ("outflow", "outflow_date"),
        "unsold_items": ("sales", "sale_date"),
    }

    if "this month" in q:
        table_info = table_map.get(intent)
        if table_info:
            latest = _latest_month_from_table(*table_info)
            result["time_filter"] = latest or datetime.now().strftime("%Y-%m")
        else:
            result["time_filter"] = datetime.now().strftime("%Y-%m")
        return result

    if "last month" in q:
        table_info = table_map.get(intent)
        latest = _latest_month_from_table(*table_info) if table_info else datetime.now().strftime("%Y-%m")
        if latest:
            result["time_filter"] = str(pd.Period(latest, freq="M") - 1)
        return result

    return result


def extract_filtered_product_phrase(question: str) -> Optional[str]:
    normalized = normalize_text(question)
    patterns = [
        r"(?:how many|number of|total count of|total number of)\s+(.+?)\s+(?:are there|do we have|are in stock|in stock|in warehouse|in the warehouse|do we currently have)",
        r"(?:how many|number of|total count of|total number of)\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        candidate = match.group(1).strip(" ?.")
        candidate = re.sub(r"\b(the|current)\b", "", candidate).strip()
        candidate = re.sub(r"\b(product|products|item|items|goods)\b", "", candidate).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if not candidate or candidate in GENERIC_PRODUCT_TERMS:
            return None
        if candidate.startswith("different "):
            candidate = candidate[len("different "):].strip()
        return _singularize_phrase(candidate)
    return None


def extract_lookup_target_phrase(question: str) -> Optional[str]:
    normalized = normalize_text(question)
    patterns = [
        r"(?:where is|where are|location of|which rack is)\s+(.+?)(?:\s+stored|\s+located|\?|$)",
        r"(?:stock level for|current stock of|how much stock is left for)\s+(.+?)(?:\?|$)",
        r"(?:show all|list all|show)\s+(.+?)(?:\s+products|\s+items|\s+goods|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate = match.group(1).strip(" ?.")
            candidate = re.sub(r"\b(the|all|warehouse)\b", "", candidate).strip()
            if candidate:
                return _singularize_phrase(candidate)
    return None


def extract_shipment_id(question: str) -> Optional[str]:
    match = re.search(r"\bship-\d+\b", question.lower())
    return match.group(0).upper() if match else None


def extract_entities(question: str, intent: str) -> dict:
    q = normalize_text(question)
    time_info = extract_time_entities(question, intent)
    entities = {
        "time_filter": time_info["time_filter"],
        "comparison_months": time_info["comparison_months"],
        "target_object": None,
        "operation": None,
        "target_phrase": None,
        "normalized_target_phrase": None,
        "search_term": None,
        "supplier_name": None,
        "location_name": None,
        "shipment_id": extract_shipment_id(question),
    }

    if any(term in q for term in ["supplier", "vendor"]):
        entities["target_object"] = "supplier"
    elif any(term in q for term in ["product", "item", "goods"]):
        entities["target_object"] = "product"
    elif "location" in q or "where" in q:
        entities["target_object"] = "location"
    elif "rack" in q:
        entities["target_object"] = "rack"
    elif "shipment" in q:
        entities["target_object"] = "shipment"
    elif any(term in q for term in ["category", "type", "kind"]):
        entities["target_object"] = "category"

    if any(term in q for term in COUNT_PHRASES):
        entities["operation"] = "count"
    elif any(term in q for term in ["total", "sum"]):
        entities["operation"] = "total"
    elif any(term in q for term in ["most", "top", "highest", "best"]):
        entities["operation"] = "top"
    elif any(term in q for term in ["compare", "comparison", "trend"]):
        entities["operation"] = "compare"
    elif any(term in q for term in ["where", "stored", "location of"]):
        entities["operation"] = "where"
    elif any(term in q for term in ["what is", "explain", "meaning of"]):
        entities["operation"] = "explain"
    elif any(term in q for term in ["show", "list"]):
        entities["operation"] = "list"

    filtered_target = extract_filtered_product_phrase(question)
    lookup_target = extract_lookup_target_phrase(question)
    if filtered_target:
        entities["target_phrase"] = filtered_target
        entities["normalized_target_phrase"] = filtered_target
        entities["search_term"] = filtered_target
    if lookup_target:
        entities["target_phrase"] = lookup_target
        entities["normalized_target_phrase"] = lookup_target
        entities["search_term"] = lookup_target

    supplier_match = re.search(r"\bfrom\s+([a-z0-9-]+)\b", q)
    if supplier_match:
        entities["supplier_name"] = supplier_match.group(1)

    location_match = re.search(r"\b(?:in|at)\s+(rack-[a-z0-9]+)\b", q)
    if location_match:
        entities["location_name"] = location_match.group(1)

    if intent == "warehouse_concept":
        for term in WAREHOUSE_TERMS:
            if term in q:
                entities["target_phrase"] = term
                entities["normalized_target_phrase"] = term
                break

    if intent == "unsold_items" and not entities["time_filter"]:
        entities["time_filter"] = datetime.now().strftime("%Y-%m")

    return entities


def _cue_score(question: str, intent: str, entities: dict) -> float:
    normalized_question = normalize_text(question)
    tokens = set(normalized_question.split())
    cues = INTENT_CUE_WORDS.get(intent, set())
    score = len(tokens & cues) / len(cues) if cues else 0.0

    if intent == "product_list":
        if any(phrase in normalized_question for phrase in ["what product are there", "list product", "show warehouse product"]):
            score = max(score, 0.95)
        if any(term in normalized_question for term in COUNT_PHRASES):
            score *= 0.4

    if intent == "product_type_summary":
        if any(phrase in normalized_question for phrase in ["what type of product", "what category of product", "what kinds of product"]):
            score = max(score, 0.95)
        if any(term in normalized_question for term in COUNT_PHRASES):
            score *= 0.5

    if intent == "product_count":
        if any(phrase in normalized_question for phrase in ["how many product", "number of product", "total product"]):
            score = max(score, 0.9)
        if entities.get("target_phrase"):
            score *= 0.3

    if intent == "filtered_product_count":
        if entities.get("target_phrase"):
            score = max(score, 0.93)
        if any(phrase in normalized_question for phrase in ["how many product", "number of product"]):
            score *= 0.25

    if intent == "filtered_product_list":
        if any(phrase in normalized_question for phrase in ["show all", "list all", "show products in", "list products from"]):
            score = max(score, 0.9)

    if intent == "sales_by_month_summary":
        if entities.get("comparison_months") or any(term in normalized_question for term in ["highest sales", "sales trend"]):
            score = max(score, 0.9)

    if intent == "shipment_status_lookup" and entities.get("shipment_id"):
        score = max(score, 0.95)

    if intent in {"location_lookup", "stock_level_lookup"} and entities.get("target_phrase"):
        score = max(score, 0.92)

    if intent == "restock_priority" and any(term in normalized_question for term in ["immediately", "priority", "first"]):
        score = max(score, 0.9)

    return score


def classify_structured_intent(question: str) -> dict:
    normalized_question = normalize_text(question)
    entities = extract_entities(question, "unsupported")

    if entities.get("shipment_id") and any(term in normalized_question for term in ["shipment", "status", "details", "lookup"]):
        intent = "shipment_status_lookup"
        similarity = 0.95
        matched_example = "what is the status of shipment ship-2001"
    elif entities.get("target_phrase") and entities.get("operation") == "where":
        intent = "location_lookup"
        similarity = 0.94
        matched_example = "where is wireless mouse stored"
    elif entities.get("target_phrase") and any(term in normalized_question for term in ["current stock of", "stock level for", "how much stock is left for"]):
        intent = "stock_level_lookup"
        similarity = 0.94
        matched_example = "what is the current stock of mechanical keyboard"
    elif any(term in normalized_question for term in ["show all", "list all", "show electronics", "show products in", "list products from"]):
        intent = "filtered_product_list"
        similarity = 0.9
        matched_example = "show all keyboards"
    elif entities.get("target_phrase") and entities.get("operation") == "count" and not any(
        phrase in normalized_question for phrase in ["how many product", "how many different product", "number of product", "total product"]
    ):
        intent = "filtered_product_count"
        similarity = 0.92
        matched_example = "how many keyboards are there in warehouse"
    else:
        intent, similarity, matched_example = match_examples(question, INTENT_EXAMPLES)
        entities = extract_entities(question, intent or "unsupported")

    cue_score = _cue_score(question, intent, entities) if intent else 0.0
    confidence = round(0.65 * similarity + 0.35 * cue_score, 3)

    if not intent or confidence < INTENT_CONFIDENCE_THRESHOLD:
        return {
            "intent": "unsupported",
            "confidence": confidence,
            "matched_example": matched_example,
            "entities": entities,
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "matched_example": matched_example,
        "entities": entities,
    }


def _format_count_answer(intent: str, df: pd.DataFrame, entities: dict) -> str:
    if df.empty:
        return "No structured data is available for this count."
    value = int(df.iloc[0, 0])
    if intent == "supplier_count":
        return f"There are {value} distinct suppliers in the current warehouse data."
    if intent == "product_count":
        return f"There are {value} distinct products in the current warehouse data."
    if intent == "location_count":
        return f"There are {value} distinct locations in the current warehouse data."
    if intent == "rack_type_count":
        return f"There are {value} distinct rack types inferred from the current location values."
    if intent == "filtered_product_count":
        target = entities.get("normalized_target_phrase") or "matching"
        return f"There are {value} {target}-related products in the current warehouse inventory."
    return f"The count is {value}."


def format_answer(intent: str, df: pd.DataFrame, entities: dict, question: str) -> str:
    q = normalize_text(question)
    month_filter = entities.get("time_filter")

    if intent in {"supplier_count", "product_count", "location_count", "rack_type_count", "filtered_product_count"}:
        return _format_count_answer(intent, df, entities)

    if intent == "supplier_list":
        if df.empty:
            return "No supplier records are available in the warehouse data."
        sample = ", ".join(df["supplier"].dropna().astype(str).head(5).tolist())
        return f"The warehouse currently works with suppliers such as {sample}."

    if intent == "product_list":
        if df.empty:
            return "No product records are available in the warehouse data."
        sample = ", ".join(df["name"].head(5).tolist())
        return f"The warehouse currently contains products such as {sample}."

    if intent == "product_type_summary":
        if df.empty:
            return "No product category data is available in the warehouse data."
        sample = ", ".join(df["category"].dropna().astype(str).head(5).tolist())
        return f"The warehouse currently contains product categories such as {sample}."

    if intent == "filtered_product_list":
        if df.empty:
            return NO_MATCHING_PRODUCTS_RESPONSE
        sample = ", ".join(df["name"].head(5).tolist())
        return f"The matching warehouse products include {sample}."

    if intent == "sales_total":
        if df.empty:
            return "No sales data matched this request."
        total_units = int(df["quantity_sold"].sum())
        total_revenue = float(df["revenue"].sum())
        label = month_filter if month_filter else "the selected period"
        top_items = ", ".join(df.sort_values("quantity_sold", ascending=False)["name"].head(3).tolist())
        return f"For {label}, total sales were {total_units} units with revenue of ${total_revenue:,.2f}. Top-selling items were: {top_items}."

    if intent == "sales_top_item":
        if df.empty:
            return "No sales data matched this request."
        row = df.iloc[0]
        label = month_filter if month_filter else "the selected period"
        return f"For {label}, the top-selling item was {row['name']} with {int(row['quantity_sold'])} units sold."

    if intent == "sales_by_month_summary":
        if df.empty:
            return "No sales data matched this request."
        if entities.get("comparison_months"):
            comparisons = ", ".join(
                f"{row['month']} ({int(row['total_quantity_sold'])} units)"
                for _, row in df.iterrows()
            )
            top_row = df.sort_values("total_quantity_sold", ascending=False).iloc[0]
            return f"Sales comparison across the selected months: {comparisons}. The highest month was {top_row['month']}."
        top_row = df.sort_values("total_quantity_sold", ascending=False).iloc[0]
        return f"The month with the highest sales was {top_row['month']} with {int(top_row['total_quantity_sold'])} units sold."

    if intent == "inflow_total":
        if df.empty:
            return "No inflow data matched this request."
        total_units = int(df["quantity_in"].sum())
        label = month_filter if month_filter else "the selected period"
        top_items = ", ".join(df.sort_values("quantity_in", ascending=False)["name"].head(3).tolist())
        return f"For {label}, total inflow was {total_units} units. Highest inflow items were: {top_items}."

    if intent == "outflow_total":
        if df.empty:
            return "No outflow data matched this request."
        total_units = int(df["quantity_out"].sum())
        label = month_filter if month_filter else "the selected period"
        top_items = ", ".join(df.sort_values("quantity_out", ascending=False)["name"].head(3).tolist())
        return f"For {label}, total outflow was {total_units} units. Highest outflow items were: {top_items}."

    if intent == "low_stock":
        if df.empty:
            return "No items are currently below reorder threshold."
        names = ", ".join(df["name"].head(10).tolist())
        return f"The following items are currently low in stock: {names}."

    if intent == "restock_priority":
        if df.empty:
            return "No items currently require restocking."
        names = ", ".join(df["name"].head(5).tolist())
        return f"Priority restock items include {names}."

    if intent == "delayed_shipments":
        if df.empty:
            return "There are no delayed shipments right now."
        suppliers = ", ".join(df["supplier"].dropna().unique().tolist())
        return f"There are {len(df)} delayed shipments. Affected suppliers include: {suppliers}."

    if intent == "shipment_status_lookup":
        if df.empty:
            return "I found no matching shipment for that shipment ID."
        row = df.iloc[0]
        return f"Shipment {row['shipment_id']} is currently marked as {row['status']} with supplier {row['supplier']} and expected date {row['expected_date']}."

    if intent == "supplier_risk":
        if df.empty:
            return "No supplier-related low-stock risk was identified."
        return f"The supplier currently affecting low stock the most is {df.iloc[0]['supplier']}."

    if intent == "category_summary":
        if df.empty:
            return "Category summary is not available."
        top_category = df.sort_values("total_quantity", ascending=False).iloc[0]["category"]
        return f"Here is the current stock summary by category. The highest-quantity category is {top_category}."

    if intent == "stock_level_lookup":
        if df.empty:
            return NO_MATCHING_PRODUCTS_RESPONSE
        row = df.iloc[0]
        return f"{row['name']} currently has {int(row['quantity'])} units in stock at {row['location']}."

    if intent == "zero_stock":
        if df.empty:
            return "No products are currently at zero stock."
        names = ", ".join(df["name"].head(10).tolist())
        return f"The following products are currently out of stock: {names}."

    if intent == "location_lookup":
        if df.empty:
            return NO_MATCHING_PRODUCTS_RESPONSE
        row = df.iloc[0]
        return f"{row['name']} is currently stored at {row['location']}."

    if intent == "location_usage_summary":
        if df.empty:
            return "No location usage data is available."
        if entities.get("location_name"):
            sample = ", ".join(df["name"].head(5).tolist())
            return f"Products in {entities['location_name']} include {sample}."
        if "empty location" in q:
            return "Empty locations cannot be determined reliably without a master location table."
        if "low-stock" in q or "low stock" in q:
            return "Use the low-stock results together with the location column to identify locations containing low-stock items."
        top_locations = ", ".join(
            f"{row['location']} ({int(row['product_count'])} products)"
            for _, row in df.head(5).iterrows()
        )
        return f"Location usage is currently led by {top_locations}."

    if intent == "unsold_items":
        if df.empty:
            return "All products were sold in the selected period."
        names = ", ".join(df["name"].head(10).tolist())
        return f"The following items were not sold in the selected period: {names}."

    if intent == "warehouse_concept":
        for term, explanation in WAREHOUSE_TERMS.items():
            if term in q:
                return explanation
        return "I can explain warehouse concepts such as SKU, inventory, inflow, outflow, shipment delay, restocking, and reorder threshold."

    return UNSUPPORTED_WAREHOUSE_RESPONSE


def answer_dynamic_question(question: str) -> Tuple[Optional[str], Optional[Dict[str, pd.DataFrame]], dict]:
    classification = classify_structured_intent(question)
    intent = classification["intent"]
    entities = classification["entities"]
    metadata = {
        "intent": intent,
        "entities": entities,
        "confidence": classification["confidence"],
        "matched_example": classification["matched_example"],
        "sql_template": "none",
    }

    if intent == "unsupported":
        return UNSUPPORTED_WAREHOUSE_RESPONSE, {}, metadata

    if intent == "warehouse_concept":
        return format_answer(intent, pd.DataFrame(), entities, question), {}, metadata

    sql, params, template_name = build_query(intent, entities)
    metadata["sql_template"] = template_name

    if intent == "filtered_product_count":
        if not sql:
            return UNSUPPORTED_WAREHOUSE_RESPONSE, {}, metadata
        df = run_sql(sql, params)
        if df.empty:
            return NO_MATCHING_PRODUCTS_RESPONSE, {"Matching Products": df}, metadata
        count_df = pd.DataFrame({"item_count": [len(df)]})
        answer = format_answer(intent, count_df, entities, question)
        return answer, {"Matching Products": df}, metadata

    if intent in {"filtered_product_list", "shipment_status_lookup", "sales_by_month_summary", "location_lookup", "stock_level_lookup", "location_usage_summary"}:
        if not sql:
            return UNSUPPORTED_WAREHOUSE_RESPONSE, {}, metadata
        df = run_sql(sql, params)
        answer = format_answer(intent, df, entities, question)
        title_map = {
            "filtered_product_list": "Matching Products",
            "shipment_status_lookup": "Shipment Status",
            "sales_by_month_summary": "Sales By Month",
            "location_lookup": "Location Lookup",
            "stock_level_lookup": "Stock Level",
            "location_usage_summary": "Location Usage",
        }
        return answer, {title_map[intent]: df}, metadata

    if intent == "unsold_items":
        if not sql:
            return UNSUPPORTED_WAREHOUSE_RESPONSE, {}, metadata
        sold_df = run_sql(sql, params)
        sold_names = sold_df["name"].tolist() if not sold_df.empty else []
        if sold_names:
            placeholders = ",".join(["?"] * len(sold_names))
            df = run_sql(
                f"SELECT sku, name, category, quantity, location, supplier FROM products WHERE name NOT IN ({placeholders})",
                tuple(sold_names),
            )
        else:
            df = run_sql("SELECT sku, name, category, quantity, location, supplier FROM products")
        answer = format_answer(intent, df, entities, question)
        return answer, {"Unsold Items": df}, metadata

    if not sql:
        return UNSUPPORTED_WAREHOUSE_RESPONSE, {}, metadata

    df = run_sql(sql, params)
    answer = format_answer(intent, df, entities, question)

    title_map = {
        "supplier_count": "Supplier Count",
        "supplier_list": "Supplier List",
        "product_count": "Product Count",
        "product_list": "Product List",
        "product_type_summary": "Product Type Summary",
        "location_count": "Location Count",
        "rack_type_count": "Rack Type Count",
        "sales_total": "Sales",
        "sales_top_item": "Sales Top Item",
        "inflow_total": "Inflow",
        "outflow_total": "Outflow",
        "low_stock": "Low Stock",
        "restock_priority": "Restock Priority",
        "delayed_shipments": "Delayed Shipments",
        "supplier_risk": "Supplier Risk",
        "category_summary": "Category Summary",
        "zero_stock": "Zero Stock",
    }

    return answer, {title_map.get(intent, "Structured Results"): df}, metadata
