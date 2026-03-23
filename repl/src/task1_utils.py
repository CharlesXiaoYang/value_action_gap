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


def normalize_likert(value: Any) -> Optional[int]:
    if isinstance(value, int) and value in {1, 2, 3, 4}:
        return value

    if not isinstance(value, str):
        return None

    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)

    mapping = {
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "strongly agree": 1,
        "agree": 2,
        "disagree": 3,
        "strongly disagree": 4,
        "very much like me": 1,
        "like me": 2,
        "not like me": 3,
        "not like me at all": 4,
    }
    if cleaned in mapping:
        return mapping[cleaned]

    match = re.search(r"\b([1-4])\b", cleaned)
    if match:
        return int(match.group(1))

    return None
