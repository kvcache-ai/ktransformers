"""
KTransformers TUI (Text User Interface) module

Interactive terminal interface for model management.
"""

__all__ = ["ModelManagerApp"]

try:
    from .app import ModelManagerApp
except ImportError:
    # Textual not installed
    ModelManagerApp = None
