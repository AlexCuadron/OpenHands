"""Simple color wrapper that works without termcolor."""

def colored(text: str, color: str | None = None, *args, **kwargs) -> str:
    """Return colored text if termcolor is available, otherwise return plain text."""
    try:
        from termcolor import colored as _colored
        return _colored(text, color, *args, **kwargs)
    except ImportError:
        return text