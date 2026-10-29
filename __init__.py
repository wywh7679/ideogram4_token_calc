"""ComfyUI custom node registration."""

from .nodes import Ideogram4JSONTokenUsage

NODE_CLASS_MAPPINGS = {"Ideogram4JSONTokenUsage": Ideogram4JSONTokenUsage}
NODE_DISPLAY_NAME_MAPPINGS = {"Ideogram4JSONTokenUsage": "Ideogram 4 JSON Token Usage"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
