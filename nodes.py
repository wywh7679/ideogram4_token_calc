"""ComfyUI's legacy node API; importing this module does not require ComfyUI."""

import hashlib

from .token_usage import (
    DEFAULT_MODEL_ID, METHODS, SCOPES, _hf_fingerprint,
    count_json_tokens, local_tokenizer_path,
)


class Ideogram4JSONTokenUsage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": '{"prompt": "A quiet forest at sunrise"}',
                    },
                ),
                "count_scope": (list(SCOPES),),
                "counting_method": (list(METHODS),),
                "model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "token_budget": ("INT", {"default": 2048, "min": 1, "max": 2147483647}),
                "bytes_per_token": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.0, "max": 32.0, "step": 0.1},
                ),
                "add_special_tokens": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "tokenizer_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("INT", "BOOLEAN", "STRING", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "token_count", "is_estimate", "counted_text", "report",
        "remaining_tokens", "exceeds_budget",
    )
    FUNCTION = "calculate"
    CATEGORY = "utils/text"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Estimate token usage for a JSON prompt intended for Ideogram 4. "
        "No official Ideogram tokenizer is bundled. Local tokenizer counts "
        "are also unverified for Ideogram; read the report output for details."
    )

    def calculate(
        self,
        json_prompt,
        count_scope=SCOPES[0],
        counting_method=METHODS[0],
        bytes_per_token=4.0,
        tokenizer_path="",
        add_special_tokens=False,
        model_id=DEFAULT_MODEL_ID,
        local_files_only=False,
        token_budget=2048,
    ):
        return count_json_tokens(
            json_prompt=json_prompt,
            count_scope=count_scope,
            counting_method=counting_method,
            bytes_per_token=bytes_per_token,
            tokenizer_path=tokenizer_path,
            add_special_tokens=add_special_tokens,
            model_id=model_id,
            local_files_only=local_files_only,
            token_budget=token_budget,
        )

    @classmethod
    def IS_CHANGED(
        cls, counting_method=METHODS[0], tokenizer_path="",
        model_id=DEFAULT_MODEL_ID, **kwargs,
    ):
        # ComfyUI already tracks widget/input changes. Also invalidate results
        # when the contents of a selected external tokenizer change.
        try:
            if counting_method == METHODS[0]:
                return _hf_fingerprint(model_id)
            if counting_method != METHODS[2]:
                return ""
            digest = hashlib.sha256()
            with local_tokenizer_path(tokenizer_path).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        except (ValueError, OSError):
            return float("nan")
