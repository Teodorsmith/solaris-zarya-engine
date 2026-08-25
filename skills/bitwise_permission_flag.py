from pathlib import Path
from typing import Dict, Any, Optional

# --- Core Functions ---

def _bit_to_mask(bit: int) -> int:
    """Convert a bit position to its corresponding bitmask."""
    return 1 << bit

def has_flag(flags: int, bit: int) -> bool:
    """Check if a flag has a specific bit set."""
    return (flags & _bit_to_mask(bit)) != 0

def set_flag(flags: int, bit: int) -> int:
    """Set a specific bit in the flags."""
    return flags | _bit_to_mask(bit)

def clear_flag(flags: int, bit: int) -> int:
    """Clear a specific bit in the flags."""
    return flags & ~_bit_to_mask(bit)

def toggle_flag(flags: int, bit: int) -> int:
    """Toggle a specific bit in the flags."""
    return flags ^ _bit_to_mask(bit)

def combine_flags(*flags_list: int) -> int:
    """Combine multiple flags using bitwise OR."""
    result = 0
    for flags in flags_list:
        result |= flags
    return result

def execute(**kwargs: Any) -> Dict[str, Any]:
    """Entry point for the skill. Handles all operations via kwargs."""
    operation = kwargs.get('operation')
    flags = kwargs.get('flags', 0)
    bit = kwargs.get('bit', 0)
    
    if operation == 'has_flag':
        return {'result': has_flag(flags, bit)}
    elif operation == 'set_flag':
        return {'result': set_flag(flags, bit)}
    elif operation == 'clear_flag':
        return {'result': clear_flag(flags, bit)}
    elif operation == 'toggle_flag':
        return {'result': toggle_flag(flags, bit)}
    elif operation == 'combine_flags':
        flags_list = kwargs.get('flags_list', [])
        return {'result': combine_flags(*flags_list)}
    else:
        return {'error': 'Invalid operation'}

# --- Helper Functions (Optional) ---

def _get_flag_names(flags: int) -> Dict[int, str]:
    """Helper to map bit positions to names (not part of core logic)."""
    return {
        0: 'READ',
        1: 'WRITE',
        2: 'EXECUTE',
        3: 'DELETE',
        4: 'ADMIN'
    }
