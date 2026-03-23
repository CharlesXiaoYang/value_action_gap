import re
from typing import Any, Dict, Optional

import json_repair


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json_repair.loads(match.group(0))
    except Exception:
        return None

    return parsed if isinstance(parsed, dict) else None


def normalize_action_choice(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    cleaned = value.strip().lower()
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned in {"1", "option 1", "option1"}:
        return "Option 1"
    if cleaned in {"2", "option 2", "option2"}:
        return "Option 2"

    match = re.search(r"option\s*([12])", cleaned)
    if match:
        return f"Option {match.group(1)}"

    if cleaned in {"positive", "pos"}:
        return "positive"
    if cleaned in {"negative", "neg"}:
        return "negative"

    return None


def resolve_selected_polarity(raw_action: Any, reverse_order: bool) -> Optional[str]:
    normalized = normalize_action_choice(raw_action)
    if normalized is None:
        return None

    if normalized in {"positive", "negative"}:
        return normalized

    option_to_polarity = {
        "Option 1": "positive" if not reverse_order else "negative",
        "Option 2": "negative" if not reverse_order else "positive",
    }
    return option_to_polarity.get(normalized)
