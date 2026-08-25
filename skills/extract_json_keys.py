import json
from typing import Any


def _recursive_extract(obj: Any, keys: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            _recursive_extract(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _recursive_extract(item, keys)


def execute(data: str | dict | list = None) -> list[str]:
    if data is None:
        return []
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            return []
    else:
        parsed = data
    keys: set[str] = set()
    _recursive_extract(parsed, keys)
    return sorted(list(keys))
