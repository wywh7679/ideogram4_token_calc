# Ideogram 4 JSON Token Usage for ComfyUI

A ComfyUI custom node that validates a JSON prompt, counts tokens with a selected tokenizer, and compares the count with a configurable budget. By default it minifies the JSON and uses `Qwen/Qwen3-VL-8B-Instruct` through Hugging Face `AutoTokenizer`.

**The Qwen tokenizer and 2,048-token default budget are user-selected assumptions, not verified Ideogram 4 specifications.** The Qwen method counts tokens exactly for the selected tokenizer and text, but its count remains a proxy for Ideogram 4. No Ideogram tokenizer is bundled, and `is_estimate` remains `True` for every method relative to Ideogram 4. Counts are not verified Ideogram usage or billing figures.

## Install

Copy this folder into `ComfyUI/custom_nodes/ideogram4_token_calc`.
After the code has been pushed to GitHub, you can alternatively run this from your ComfyUI directory:

```sh
git clone https://github.com/wywh7679/ideogram4_token_calc.git custom_nodes/ideogram4_token_calc
```

Install the dependencies with the same Python interpreter that runs ComfyUI:

```sh
python -m pip install -r custom_nodes/ideogram4_token_calc/requirements.txt
```

For Windows portable, run this from the directory containing `python_embeded` and `ComfyUI`:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ideogram4_token_calc\requirements.txt
```

Restart ComfyUI and add **Ideogram 4 JSON Token Usage** from **utils/text**. The dependencies are `transformers==4.57.1` and `tokenizers==0.22.1`. The default method downloads tokenizer assets, not model weights. The byte-estimate method can run with Python's standard library alone.

## Use

Paste valid JSON into `json_prompt`, for example:

```json
{
  "subject": "A red bicycle beside a stone wall",
  "style": "Watercolor",
  "details": ["Soft morning light", "Muted colors"]
}
```

Select what to count with `count_scope`:

- **JSON text (minified)** is the default. After validating the JSON, it removes insignificant whitespace outside string literals. It preserves string escapes, number spellings, duplicate key values, and key order. Keys and punctuation are counted.
- **JSON text (as entered)** counts the entire input, including keys, punctuation, escapes, indentation, and whitespace. Use this when the JSON itself is the prompt text.
- **String values only** traverses objects and arrays in input order, omits keys and non-string values, and joins string values with newline characters. Those added newlines affect the count. JSON containing no string values produces empty counted text and a count of zero in estimate mode.

Select `counting_method`:

- **Hugging Face tokenizer** is the default. It loads `model_id` with `AutoTokenizer.from_pretrained(..., trust_remote_code=False)` and counts the selected text. The default model is `Qwen/Qwen3-VL-8B-Instruct`; you can select another supported model or a local tokenizer directory. Loaded tokenizers are cached. Set `local_files_only=True` to require assets already in the local Hugging Face cache or directory. Initial downloads require access to `huggingface.co` and any hosts to which its asset requests redirect.
- **Estimate (UTF-8 bytes)** returns `ceil(UTF-8 byte count / bytes_per_token)`, or zero for empty counted text. The default ratio is `4`; adjust it if you have measured a more suitable ratio. This is a rough heuristic with no accuracy guarantee, especially across languages.
- **Local tokenizer.json** reads the file specified in `tokenizer_path`. Use a path accessible to the ComfyUI process, preferably absolute. Enable `add_special_tokens` only when the tokenizer and intended workflow require them. Truncation and padding are disabled so they cannot hide the full token count. No tokenizer download or API request is made. A supplied tokenizer is not automatically identified or verified as Ideogram 4's tokenizer.

`add_special_tokens` defaults to `False`. No chat template, image or video tokens, or API-wrapper overhead is included. Enable special tokens only when they match the intended workflow. No text is truncated to fit the budget.

Send `counted_text` unchanged as the actual prompt if you want the selected tokenizer's count to match what you send. For example, counting minified JSON and then sending the original indented JSON can produce a different count.

Set `token_budget` to the limit you want to compare against; the default is `2048`. The node reports signed `remaining_tokens = token_budget - token_count` and whether the count exceeds that budget. A nonnegative result does not establish that Ideogram will accept the prompt, that generation is safe from truncation, or that its actual usage is within a documented limit.

If you obtain an official tokenizer file, you can supply it locally, but matching Ideogram's preprocessing and special-token behavior is also necessary to establish an exact model count.

Malformed JSON fails with an error instead of returning a misleading count. ComfyUI dynamic prompt expansion is disabled for this input so that JSON text is preserved.

## Outputs

| Output | Meaning |
| --- | --- |
| `token_count` | Integer count from the selected method. |
| `is_estimate` | Always `True` relative to Ideogram 4, whose tokenizer and preprocessing are unverified. |
| `counted_text` | The exact text passed to the selected counting method. |
| `report` | JSON report describing the method, counts, and verification limitation. |
| `remaining_tokens` | Signed difference between the configured budget and count; negative means over budget. |
| `exceeds_budget` | `True` when `token_count` is greater than `token_budget`. |

The report includes `token_budget`, `remaining_tokens`, `exceeds_budget`, and `model_tokenizer_verified: false`; the Hugging Face method also identifies the selected tokenizer.

The node executes when queued even without downstream connections. Connect `report` or `counted_text` to a text-display node to view it; this package does not add an automatic text display.

## Development

Run the tests from this repository:

```sh
python -m unittest discover -s tests -v
```

Install `requirements.txt` to exercise tokenizer support. The node exports the standard `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` ComfyUI registration dictionaries.
