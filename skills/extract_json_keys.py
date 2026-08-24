import json
from typing import Any, List, Set, Union

def _recursive_extract(obj: Any, keys: Set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            _recursive_extract(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _recursive_extract(item, keys)

def execute(data: Union[str, dict, list] = None) -> List[str]:
    if data is None:
        return []
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            return []
    else:
        parsed = data
    keys: Set[str] = set()
    _recursive_extract(parsed, keys)
    return sorted(list(keys))