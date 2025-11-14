"""Utility functions for label management"""

from db.models import Label


def is_human_labeled(label: Label | None) -> bool:
    """
    Check if label is created by human (vs auto/machine)
    
    Args:
        label: Label object or None
    
    Returns:
        True if label is human-created, False otherwise
    
    Logic:
        - No label -> False
        - Empty label_data -> False
        - created_by starts with "auto" -> False (machine-generated)
        - Otherwise -> True (human-generated)
    """
    if not label:
        return False
    if not label.label_data or len(label.label_data) == 0:
        return False
    if label.created_by.startswith("auto"):
        return False
    return True

