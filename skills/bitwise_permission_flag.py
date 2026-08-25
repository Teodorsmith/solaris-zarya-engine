from pathlib import Path
from typing import Union


def has_flag(flags: int, bit: int) -> bool:
    """Check if a specific bit is set in the flags."""
    return (flags & (1 << bit)) != 0

def set_flag(flags: int, bit: int) -> int:
    """Set a specific bit in the flags."""
    return flags | (1 << bit)

def clear_flag(flags: int, bit: int) -> int:
    """Clear a specific bit in the flags."""
    return flags & ~(1 << bit)

def toggle_flag(flags: int, bit: int) -> int:
    """Toggle a specific bit in the flags."""
    return flags ^ (1 << bit)

def execute(flags: int = 0, bit: int = 0, operation: str = "check") -> dict:
    """Execute the specified operation on the flags."""
    operations = {
        "check": lambda: has_flag(flags, bit),
        "set": lambda: set_flag(flags, bit),
        "clear": lambda: clear_flag(flags, bit),
        "toggle": lambda: toggle_flag(flags, bit)
    }
    
    if operation not in operations:
        return {"error": "Invalid operation"}
    
    result = operations[operation]()
    return {
        "operation": operation,
        "flags": flags,
        "bit": bit,
        "result": result
    }