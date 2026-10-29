"""JSON prompt counting with local or Hugging Face tokenizers and an estimate mode."""

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re


SCOPES = ("JSON text (minified)", "JSON text (as entered)", "String values only")
METHODS = ("Hugging Face tokenizer", "Estimate (UTF-8 bytes)", "Local tokenizer.json")
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Match complete JSON string literals before matching whitespace so escaped
# quotes, spaces within strings, numbers, and duplicate keys stay unchanged.
_JSON_WHITESPACE = re.compile(r'("(?:\\.|[^"\\])*")|\s+')


def _minify_json(source):
    return _JSON_WHITESPACE.sub(lambda match: match.group(1) or "", source)


def _hf_fingerprint(model_id):
    """Track local tokenizer changes without hashing any model weights."""
    directory = Path(model_id).expanduser()
    if not directory.is_dir():
        return ""
    names = (
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
        "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "config.json",
    )
    digest = hashlib.sha256()
    for name in names:
        path = directory / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_hf_tokenizer(model_id, local_files_only, fingerprint):
    # fingerprint participates in the cache key for local directories.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=False, local_files_only=local_files_only,
    )


def _hf_count(text, model_id, local_files_only, add_special_tokens):
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Hugging Face tokenizer mode requires a model_id or local directory.")
    try:
        model_path = Path(model_id).expanduser()
        identifier = str(model_path.resolve()) if model_path.is_dir() else model_id
        tokenizer = _load_hf_tokenizer(
            identifier, bool(local_files_only), _hf_fingerprint(identifier)
        )
        tokens = tokenizer.encode(
            text, add_special_tokens=bool(add_special_tokens), truncation=False,
        )
        return len(tokens)
    except ImportError as exc:
        raise ValueError(
            "Hugging Face tokenizer mode requires 'transformers'. Install "
            "requirements.txt using ComfyUI's Python interpreter."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Could not load or encode with tokenizer '{model_id}': {exc}. "
            "Check Hugging Face access, pre-cache the tokenizer, or supply a local "
            "tokenizer directory. Remote custom code is disabled."
        ) from exc


class _ObjectPairs(list):
    """Keep duplicate object values and document order during text extraction."""


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}. Use valid JSON values.")


def _parse_json(json_prompt):
    if not isinstance(json_prompt, str) or not json_prompt.strip():
        raise ValueError("Enter a JSON prompt before calculating token usage.")
    try:
        return json.loads(
            json_prompt,
            object_pairs_hook=_ObjectPairs,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except RecursionError as exc:
        raise ValueError("JSON prompt nesting exceeds the parser's supported depth.") from exc


def _string_values(value):
    strings = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, _ObjectPairs):
            pending.extend(value for _, value in reversed(item))
        elif isinstance(item, list):
            pending.extend(reversed(item))
    return "\n".join(strings)


def local_tokenizer_path(tokenizer_path):
    if not isinstance(tokenizer_path, str) or not tokenizer_path.strip():
        raise ValueError("Local tokenizer.json mode requires a tokenizer_path.")
    path = Path(tokenizer_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Tokenizer file does not exist or is not a file: {path}")
    return path


def _local_count(text, tokenizer_path, add_special_tokens):
    path = local_tokenizer_path(tokenizer_path)
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ValueError(
            "Local tokenizer.json mode requires the optional 'tokenizers' package. "
            "Install requirements.txt using ComfyUI's Python interpreter."
        ) from exc
    try:
        tokenizer = Tokenizer.from_file(str(path))
        # A tokenizer exported with inference limits must not silently truncate
        # or pad a prompt when measuring its token usage.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)
    except Exception as exc:
        raise ValueError(f"Could not count with the local tokenizer file: {exc}") from exc


def count_json_tokens(
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
    """Return count, estimate flag, text, report, remaining tokens, and excess flag.

    A local tokenizer yields a precise count for that tokenizer configuration,
    but remains an unverified estimate of Ideogram 4's actual usage.
    """
    if count_scope not in SCOPES:
        raise ValueError(f"Unknown count_scope: {count_scope}")
    if counting_method not in METHODS:
        raise ValueError(f"Unknown counting_method: {counting_method}")
    if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < 1:
        raise ValueError("token_budget must be a positive integer.")
    parsed = _parse_json(json_prompt)
    if count_scope == SCOPES[0]:
        text = _minify_json(json_prompt)
    elif count_scope == SCOPES[1]:
        text = json_prompt
    else:
        text = _string_values(parsed)
    try:
        byte_count = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError("Counted text contains an unpaired Unicode surrogate.") from exc

    if counting_method == METHODS[1]:
        if (
            isinstance(bytes_per_token, bool)
            or not isinstance(bytes_per_token, (int, float))
            or not math.isfinite(bytes_per_token)
            or not 1.0 <= bytes_per_token <= 32.0
        ):
            raise ValueError("bytes_per_token must be a finite number from 1 to 32.")
        count = math.ceil(byte_count / bytes_per_token)
        warning = (
            "Heuristic estimate: UTF-8 byte count divided by bytes_per_token, "
            "rounded up. This is not measured Ideogram 4 token usage and has "
            "no guaranteed error bound. Do not use it as a billing count."
        )
    elif counting_method == METHODS[2]:
        count = _local_count(text, tokenizer_path, add_special_tokens)
        warning = (
            "Count measured with the supplied local tokenizer, with truncation "
            "and padding disabled. Its vocabulary and preprocessing have not "
            "been verified against Ideogram 4. This is not verified Ideogram 4 "
            "usage or a billing count."
        )
    else:
        count = _hf_count(text, model_id, local_files_only, add_special_tokens)
        warning = (
            "Exact count for the selected Hugging Face tokenizer and counted text. "
            "Qwen as Ideogram 4's native text encoder has not been verified. "
            "No chat template, image tokens, or service-added tokens are included. "
            "This is not verified Ideogram 4 usage or a billing count."
        )

    warning += (
        " The configured budget is user-supplied; 2048 is not a verified Ideogram 4 "
        "limit. Being within budget does not guarantee API acceptance or generation quality."
    )
    remaining = token_budget - count

    report = {
        "model": "Ideogram 4",
        "token_count": count,
        "is_estimate": True,
        "model_tokenizer_verified": False,
        "counting_method": counting_method,
        "count_scope": count_scope,
        "character_count": len(text),
        "utf8_byte_count": byte_count,
        "bytes_per_token": bytes_per_token if counting_method == METHODS[1] else None,
        "add_special_tokens": bool(add_special_tokens) if counting_method != METHODS[1] else False,
        "model_id": model_id if counting_method == METHODS[0] else None,
        "local_files_only": bool(local_files_only) if counting_method == METHODS[0] else None,
        "token_budget": token_budget,
        "remaining_tokens": remaining,
        "exceeds_budget": remaining < 0,
        "warning": warning,
    }
    return count, True, text, json.dumps(report, ensure_ascii=False, indent=2), remaining, remaining < 0
