import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

SYNONYM_MAP = {
    "vendors": "supplier",
    "vendor": "supplier",
    "suppliers": "supplier",
    "items": "product",
    "item": "product",
    "goods": "product",
    "products": "product",
    "kinds": "types",
    "categories": "category",
    "types": "type",
    "available": "there",
    "sold": "sales",
    "revenue": "sales",
    "reorder": "restock",
    "replenish": "restock",
    "replenishment": "restock",
    "low stock": "restock",
    "late shipment": "delayed shipment",
    "shipment delay": "delayed shipment",
    "delayed shipments": "delayed shipment",
    "incoming stock": "inflow",
    "stock received": "inflow",
    "received stock": "inflow",
    "outgoing stock": "outflow",
    "dispatched stock": "outflow",
    "storage points": "location",
    "storage locations": "location",
    "locations": "location",
    "racks": "rack",
    "late shipments": "delayed shipment",
    "vendors": "supplier",
    "replenish": "restock",
    "replenishment": "restock",
    "kinds": "types",
}


def normalize_text(text: str) -> str:
    normalized = text.lower().strip()
    for source, target in sorted(SYNONYM_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    return [token for token in normalized.split() if token]


def _token_overlap_score(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def similarity_score(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    sequence_score = SequenceMatcher(None, left_norm, right_norm).ratio()
    overlap_score = _token_overlap_score(tokenize(left_norm), tokenize(right_norm))
    return 0.55 * sequence_score + 0.45 * overlap_score


def match_examples(question: str, examples_by_label: Dict[str, List[str]]) -> Tuple[str, float, str]:
    best_label = ""
    best_score = 0.0
    best_example = ""

    for label, examples in examples_by_label.items():
        for example in examples:
            score = similarity_score(question, example)
            if score > best_score:
                best_label = label
                best_score = score
                best_example = example

    return best_label, best_score, best_example
