"""
Logging utilities for secure logging with sensitive data sanitization.
"""
from typing import Dict, Any


SENSITIVE_KEYS = {'api_key', 'password', 'token', 'secret', 'authorization'}


def sanitize_dict(data: Dict[str, Any], mask: str = '***REDACTED***') -> Dict[str, Any]:
    """
    Sanitize a dictionary by masking sensitive fields.

    Args:
        data: Dictionary to sanitize
        mask: Mask string to use for sensitive values

    Returns:
        New dictionary with sensitive values masked

    Example:
        >>> sanitize_dict({'api_key': 'secret', 'name': 'test'})
        {'api_key': '***REDACTED***', 'name': 'test'}
    """
    if not isinstance(data, dict):
        return data

    sanitized = {}
    for key, value in data.items():
        if key.lower() in SENSITIVE_KEYS:
            sanitized[key] = mask
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, mask)
        else:
            sanitized[key] = value

    return sanitized


def sanitize_for_logging(obj: Any) -> str:
    """
    Convert an object to a safe string for logging.

    Args:
        obj: Object to convert to string

    Returns:
        Safe string representation with sensitive data masked
    """
    if isinstance(obj, dict):
        sanitized = sanitize_dict(obj)
        return str(sanitized)
    return str(obj)
