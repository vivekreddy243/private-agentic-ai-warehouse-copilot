from typing import Dict, List

from app.services.local_semantic_matcher import normalize_text, similarity_score

OUT_OF_SCOPE_RESPONSE = "I’m designed for warehouse operations queries only and do not have personal or general-world information."

STRUCTURED_KEYWORDS = {
    "inventory", "product", "stock", "restock", "supplier", "shipment", "sales",
    "inflow", "outflow", "warehouse", "rack", "location", "sku", "category",
    "threshold", "reorder", "quantity", "vendor", "goods", "stored", "trend",
    "shipment id", "out of stock", "stock level",
}

DOCUMENT_KEYWORDS = {
    "process", "policy", "procedure", "guideline", "guidelines", "steps", "sop",
    "damaged goods", "returns", "audit", "what should staff do", "instruction",
}

OUT_OF_SCOPE_KEYWORDS = {
    "my name", "weather", "joke", "sports", "match", "score", "who are you",
    "who won", "movie", "music", "news", "birthday",
}

ROUTE_EXAMPLES: Dict[str, List[str]] = {
    "structured": [
        "how many suppliers are there in this warehouse",
        "how many products do we have",
        "show all scanners",
        "what products are in rack-a1",
        "which month has highest sales",
        "what is the current stock of mechanical keyboard",
        "what is the inflow this month",
        "what is the outflow this month",
        "which shipments are delayed",
        "show stock by category",
        "what is the most selling item in the warehouse",
        "which items should I restock immediately",
    ],
    "document": [
        "what is the damaged goods process",
        "what is the returns policy",
        "what is supplier delay procedure",
        "what is the inventory audit guideline",
        "what should staff do for damaged goods",
    ],
    "out_of_scope": [
        "what is my name",
        "who are you",
        "what is the weather",
        "tell me a joke",
        "who won the match",
    ],
}


def _keyword_score(question: str, keywords: set[str]) -> float:
    question_norm = normalize_text(question)
    matched = [keyword for keyword in keywords if keyword in question_norm]
    if not matched:
        return 0.0
    return min(1.0, 0.35 + 0.15 * len(matched))


def _best_example_score(question: str, examples: List[str]) -> tuple[float, str]:
    best_score = 0.0
    best_example = ""
    for example in examples:
        score = similarity_score(question, example)
        if score > best_score:
            best_score = score
            best_example = example
    return best_score, best_example


def classify_route(question: str) -> dict:
    per_route_example = {
        route: _best_example_score(question, examples)
        for route, examples in ROUTE_EXAMPLES.items()
    }

    route_scores = {
        "structured": 0.55 * per_route_example["structured"][0] + 0.45 * _keyword_score(question, STRUCTURED_KEYWORDS),
        "document": 0.55 * per_route_example["document"][0] + 0.45 * _keyword_score(question, DOCUMENT_KEYWORDS),
        "out_of_scope": 0.55 * per_route_example["out_of_scope"][0] + 0.45 * _keyword_score(question, OUT_OF_SCOPE_KEYWORDS),
    }

    best_route = max(route_scores, key=route_scores.get)
    confidence = route_scores[best_route]

    if confidence < 0.35:
        best_route = "out_of_scope"

    return {
        "route": best_route,
        "confidence": round(confidence, 3),
        "matched_example": per_route_example[best_route][1],
        "scores": route_scores,
    }


def is_document_question(question: str) -> bool:
    return classify_route(question)["route"] == "document"


def is_warehouse_question(question: str) -> bool:
    route = classify_route(question)["route"]
    return route in {"structured", "document"}


def get_out_of_scope_response() -> str:
    return OUT_OF_SCOPE_RESPONSE
