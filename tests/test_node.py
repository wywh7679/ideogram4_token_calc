"""Behavioral tests that run without a ComfyUI installation or network access."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "ideogram4_token_calc_under_test"
spec = importlib.util.spec_from_file_location(
    PACKAGE_NAME,
    REPOSITORY / "__init__.py",
    submodule_search_locations=[str(REPOSITORY)],
)
package = importlib.util.module_from_spec(spec)
sys.modules[PACKAGE_NAME] = package
spec.loader.exec_module(package)
usage_module = sys.modules[f"{PACKAGE_NAME}.token_usage"]
count_json_tokens = usage_module.count_json_tokens
Node = package.NODE_CLASS_MAPPINGS["Ideogram4JSONTokenUsage"]

MINIFIED_SCOPE = "JSON text (minified)"
RAW_SCOPE = "JSON text (as entered)"
VALUE_SCOPE = "String values only"
ESTIMATE = "Estimate (UTF-8 bytes)"
LOCAL = "Local tokenizer.json"
HF = "Hugging Face tokenizer"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def estimate_tokens(json_prompt, **kwargs):
    """Keep estimator tests explicit as the production default is now HF."""
    kwargs.setdefault("counting_method", ESTIMATE)
    kwargs.setdefault("count_scope", RAW_SCOPE)
    return count_json_tokens(json_prompt, **kwargs)


class EstimateTests(unittest.TestCase):
    def test_raw_json_preserves_whitespace_keys_and_escapes(self):
        source = ' \n {"prompt": "red\\nbicycle", "steps": 20} \t'
        count, estimated, counted, report_text, _, _ = estimate_tokens(source)
        self.assertEqual(counted, source)
        self.assertEqual(count, math.ceil(len(source.encode("utf-8")) / 4))
        self.assertTrue(estimated)
        report = json.loads(report_text)
        self.assertEqual(report["count_scope"], RAW_SCOPE)
        self.assertEqual(report["character_count"], len(source))
        self.assertEqual(report["utf8_byte_count"], len(source.encode("utf-8")))

    def test_utf8_bytes_and_rounding_are_used_for_unicode(self):
        count, estimated, counted, _, _, _ = estimate_tokens('"猫🐈"')
        self.assertEqual(counted, '"猫🐈"')
        self.assertEqual(len(counted), 4)
        self.assertEqual(count, 3)  # Nine UTF-8 bytes divided by four, rounded up.
        self.assertTrue(estimated)

    def test_ratio_is_configurable(self):
        count, _, _, _, _, _ = estimate_tokens('"abc"', bytes_per_token=2)
        self.assertEqual(count, 3)

    def test_nested_values_follow_document_order_and_omit_keys(self):
        source = '{"ignore this key": ["red", {"style": "ink", "n": 2}], "tail": "bike"}'
        count, _, counted, _, _, _ = estimate_tokens(source, count_scope=VALUE_SCOPE)
        self.assertEqual(counted, "red\nink\nbike")
        self.assertEqual(count, 3)

    def test_duplicate_object_keys_do_not_discard_prompt_values(self):
        _, _, counted, _, _, _ = estimate_tokens(
            '{"prompt": "first", "prompt": "second"}', count_scope=VALUE_SCOPE
        )
        self.assertEqual(counted, "first\nsecond")

    def test_scalar_string_is_decoded_in_value_scope(self):
        _, _, counted, _, _, _ = estimate_tokens('"red\\nbike"', count_scope=VALUE_SCOPE)
        self.assertEqual(counted, "red\nbike")

    def test_no_string_values_produces_zero_estimate(self):
        for source in ('null', '123', '[false, true, 2]', '{}', '[]', '""'):
            with self.subTest(source=source):
                count, estimated, counted, _, _, _ = estimate_tokens(
                    source, count_scope=VALUE_SCOPE
                )
                self.assertEqual((count, estimated, counted), (0, True, ""))

    def test_empty_strings_keep_join_separators(self):
        count, _, counted, _, _, _ = estimate_tokens('["", ""]', count_scope=VALUE_SCOPE)
        self.assertEqual(counted, "\n")
        self.assertEqual(count, 1)

    def test_report_identifies_estimate_and_unverified_model(self):
        count, estimated, _, report_text, _, _ = estimate_tokens('{"prompt":"bike"}')
        report = json.loads(report_text)
        self.assertEqual(report["model"], "Ideogram 4")
        self.assertEqual(report["token_count"], count)
        self.assertTrue(report["is_estimate"])
        self.assertTrue(estimated)
        self.assertFalse(report["model_tokenizer_verified"])
        self.assertEqual(report["counting_method"], ESTIMATE)
        self.assertFalse(report["add_special_tokens"])
        self.assertTrue(report["warning"])

    def test_estimation_needs_neither_tokenizers_nor_network(self):
        with patch.dict(sys.modules, {"tokenizers": None}), patch(
            "socket.create_connection", side_effect=AssertionError("Network not permitted")
        ):
            count, _, _, _, _, _ = estimate_tokens('"bike"')
        self.assertEqual(count, 2)


class InputValidationTests(unittest.TestCase):
    def test_malformed_json_has_line_and_column(self):
        with self.assertRaises(ValueError) as caught:
            estimate_tokens('{\n  "prompt":\n}')
        message = str(caught.exception).lower()
        self.assertIn("line", message)
        self.assertIn("column", message)

    def test_empty_and_trailing_json_are_rejected(self):
        for source in ("", "   ", '{} {}', '{"a": 1,}'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                estimate_tokens(source)

    def test_non_json_numeric_constants_are_rejected(self):
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant), self.assertRaises(ValueError):
                estimate_tokens('{"value": ' + constant + '}')

    def test_invalid_estimate_ratios_are_rejected(self):
        for ratio in (0, -1, 33, float("nan"), float("inf"), float("-inf")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                estimate_tokens('"bike"', bytes_per_token=ratio)

    def test_unknown_scope_and_method_are_rejected(self):
        for kwargs in ({"count_scope": "unknown"}, {"counting_method": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_tokens('{}', **kwargs)

    def test_local_method_requires_an_existing_file(self):
        for path in ("", str(REPOSITORY / "does-not-exist-tokenizer.json")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                estimate_tokens('"bike"', counting_method=LOCAL, tokenizer_path=path)

    def test_missing_optional_dependency_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_path = Path(directory) / "tokenizer.json"
            tokenizer_path.write_text('{}', encoding="utf-8")
            with patch.dict(sys.modules, {"tokenizers": None}):
                with self.assertRaises(ValueError) as caught:
                    estimate_tokens(
                        '"bike"', counting_method=LOCAL, tokenizer_path=str(tokenizer_path)
                    )
        self.assertIn("tokenizers", str(caught.exception).lower())


class MinifiedJSONTests(unittest.TestCase):
    def test_default_scope_removes_only_whitespace_outside_strings(self):
        source = ' \n { "prompt" : "red  bicycle\\n猫", "n" : 2 } \t'
        result = count_json_tokens(source, counting_method=ESTIMATE)
        self.assertEqual(result[2], '{"prompt":"red  bicycle\\n猫","n":2}')
        self.assertEqual(json.loads(result[3])["count_scope"], MINIFIED_SCOPE)

    def test_minification_preserves_duplicate_keys_and_number_spellings(self):
        source = ' { "x":1.2300, "x":1e999, "n":9007199254740993 } '
        result = count_json_tokens(source, counting_method=ESTIMATE)
        self.assertEqual(result[2], '{"x":1.2300,"x":1e999,"n":9007199254740993}')

    def test_minification_preserves_escapes_and_escaped_quotes(self):
        source = r' { "prompt" : "a \"quote\" and \\ and \u732b" } '
        result = count_json_tokens(source, counting_method=ESTIMATE)
        self.assertEqual(result[2], r'{"prompt":"a \"quote\" and \\ and \u732b"}')

    def test_minification_still_validates_json(self):
        with self.assertRaises(ValueError):
            count_json_tokens('{ "a" : 1 2 }', counting_method=ESTIMATE)


class TokenBudgetTests(unittest.TestCase):
    def test_budget_below_at_and_above_usage(self):
        for budget, remaining, exceeded in ((2, -1, True), (3, 0, False), (4, 1, False)):
            with self.subTest(budget=budget):
                result = count_json_tokens(
                    '"abcdefghij"', counting_method=ESTIMATE, token_budget=budget
                )
                self.assertEqual(result[0], 3)
                self.assertEqual(result[4:6], (remaining, exceeded))
                report = json.loads(result[3])
                self.assertEqual(report["token_budget"], budget)
                self.assertEqual(report["remaining_tokens"], remaining)
                self.assertEqual(report["exceeds_budget"], exceeded)

    def test_budget_must_be_a_positive_integer(self):
        for budget in (0, -1, 1.5, True, "2048", None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                count_json_tokens('{}', counting_method=ESTIMATE, token_budget=budget)


class ComfyNodeTests(unittest.TestCase):
    def test_package_exports_node_and_display_name(self):
        self.assertEqual(Node.__name__, "Ideogram4JSONTokenUsage")
        self.assertEqual(
            package.NODE_DISPLAY_NAME_MAPPINGS["Ideogram4JSONTokenUsage"],
            "Ideogram 4 JSON Token Usage",
        )
        self.assertTrue(Node.OUTPUT_NODE)
        self.assertEqual(Node.RETURN_TYPES, ("INT", "BOOLEAN", "STRING", "STRING", "INT", "BOOLEAN"))
        self.assertEqual(
            Node.RETURN_NAMES, ("token_count", "is_estimate", "counted_text", "report", "remaining_tokens", "exceeds_budget")
        )

    def test_json_widget_preserves_literal_prompt_syntax(self):
        widget_type, options = Node.INPUT_TYPES()["required"]["json_prompt"]
        self.assertEqual(widget_type, "STRING")
        self.assertTrue(options["multiline"])
        self.assertFalse(options["dynamicPrompts"])

    def test_registered_function_executes_with_json_input(self):
        node = Node()
        result = getattr(node, Node.FUNCTION)(
            '{"prompt":"red bicycle"}', count_scope=VALUE_SCOPE, counting_method=ESTIMATE
        )
        self.assertEqual(result[0:3], (3, True, "red bicycle"))
        self.assertEqual(json.loads(result[3])["token_count"], 3)

    def test_estimation_does_not_depend_on_local_file(self):
        self.assertEqual(Node.IS_CHANGED(counting_method=ESTIMATE), "")

    def test_tokenizer_content_changes_invalidate_node_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_bytes(b'{"version":1}')
            before = Node.IS_CHANGED(counting_method=LOCAL, tokenizer_path=str(path))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_bytes(b'{"version":2}')
            after = Node.IS_CHANGED(counting_method=LOCAL, tokenizer_path=str(path))
            self.assertNotEqual(before, after)

    def test_missing_local_file_invalidates_cache(self):
        changed = Node.IS_CHANGED(counting_method=LOCAL, tokenizer_path="")
        self.assertTrue(math.isnan(changed))

    def test_widgets_default_to_qwen_minified_json_and_2048_budget(self):
        inputs = Node.INPUT_TYPES()
        widgets = {**inputs.get("required", {}), **inputs.get("optional", {})}
        self.assertEqual(widgets["count_scope"][0][0], MINIFIED_SCOPE)
        self.assertEqual(widgets["counting_method"][0][0], HF)
        self.assertEqual(widgets["model_id"][1]["default"], DEFAULT_MODEL)
        self.assertEqual(widgets["token_budget"][1]["default"], 2048)


try:
    import tokenizers
except ImportError:
    tokenizers = None

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizerFast = None


class HuggingFaceDependencyTests(unittest.TestCase):
    def test_missing_transformers_dependency_has_actionable_error(self):
        with patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises(ValueError) as caught:
                count_json_tokens('{}', model_id="missing/dependency-test-model")
        self.assertIn("transformers", str(caught.exception).lower())


@unittest.skipIf(AutoTokenizer is None, "Install transformers for Hugging Face tests")
class HuggingFaceMockTests(unittest.TestCase):
    def setUp(self):
        usage_module._load_hf_tokenizer.cache_clear()
        self.addCleanup(usage_module._load_hf_tokenizer.cache_clear)

    def test_default_model_and_safe_load_flags_and_untruncated_encoding(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [10, 20, 30]
        with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer) as load:
            result = count_json_tokens(' { "prompt" : "red bicycle" } ')
        args, kwargs = load.call_args
        self.assertEqual(args[0], DEFAULT_MODEL)
        self.assertIs(kwargs["trust_remote_code"], False)
        self.assertIs(kwargs["local_files_only"], False)
        tokenizer.encode.assert_called_once_with(
            '{"prompt":"red bicycle"}', add_special_tokens=False, truncation=False
        )
        self.assertEqual(result[0], 3)
        self.assertEqual(result[4:6], (2045, False))
        self.assertEqual(json.loads(result[3])["model_id"], DEFAULT_MODEL)

    def test_offline_mode_and_special_tokens_are_forwarded(self):
        tokenizer = Mock()
        tokenizer.encode.return_value = [1, 2]
        with patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer) as load:
            result = count_json_tokens(
                '"bike"',
                model_id="test/offline-options",
                local_files_only=True,
                add_special_tokens=True,
            )
        self.assertIs(load.call_args.kwargs["local_files_only"], True)
        tokenizer.encode.assert_called_once_with(
            '"bike"', add_special_tokens=True, truncation=False
        )
        self.assertEqual(result[0], 2)
        self.assertTrue(json.loads(result[3])["local_files_only"])

    def test_model_load_failure_explains_model_and_next_action(self):
        with patch.object(AutoTokenizer, "from_pretrained", side_effect=OSError("offline")):
            with self.assertRaises(ValueError) as caught:
                count_json_tokens('{}', model_id="test/unavailable-model")
        message = str(caught.exception).lower()
        self.assertIn("test/unavailable-model", message)
        self.assertTrue(any(word in message for word in ("download", "cache", "network", "local")))


@unittest.skipIf(
    AutoTokenizer is None or tokenizers is None,
    "Install transformers and tokenizers for local Hugging Face tests",
)
class HuggingFaceLocalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name)
        backend = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"[UNK]": 0, "[BOS]": 1, "red": 2, "bicycle": 3}, unk_token="[UNK]"
            )
        )
        backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        backend.post_processor = tokenizers.processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)]
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", bos_token="[BOS]"
        )
        tokenizer.save_pretrained(str(self.model_path))

    def count(self, **kwargs):
        return count_json_tokens(
            '{"prompt":"red bicycle"}',
            count_scope=VALUE_SCOPE,
            counting_method=HF,
            model_id=str(self.model_path),
            local_files_only=True,
            **kwargs,
        )

    def test_auto_tokenizer_loads_local_fixture_without_network(self):
        with patch("socket.create_connection", side_effect=AssertionError("No network")):
            result = self.count(token_budget=1)
        self.assertEqual(result[0:3], (2, True, "red bicycle"))
        self.assertEqual(result[4:6], (-1, True))
        report = json.loads(result[3])
        self.assertEqual(report["counting_method"], HF)
        self.assertFalse(report["model_tokenizer_verified"])

    def test_auto_tokenizer_special_token_option_changes_usage(self):
        self.assertEqual(self.count(add_special_tokens=False)[0], 2)
        self.assertEqual(self.count(add_special_tokens=True)[0], 3)


@unittest.skipIf(tokenizers is None, "Install tokenizer dependencies for local tokenizer tests")
class LocalTokenizerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tokenizer_path = Path(self.directory.name) / "tokenizer.json"
        tokenizer = tokenizers.Tokenizer(
            tokenizers.models.WordLevel(
                {"[UNK]": 0, "[BOS]": 1, "red": 2, "bicycle": 3}, unk_token="[UNK]"
            )
        )
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
        tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)]
        )
        tokenizer.save(str(self.tokenizer_path))

    def count(self, **kwargs):
        return count_json_tokens(
            '{"prompt":"red bicycle"}',
            count_scope=VALUE_SCOPE,
            counting_method=LOCAL,
            tokenizer_path=str(self.tokenizer_path),
            **kwargs,
        )

    def test_local_tokenizer_counts_ids_without_network(self):
        with patch("socket.create_connection", side_effect=AssertionError("No network")):
            count, estimated, counted, report_text, _, _ = self.count()
        self.assertEqual(count, 2)
        self.assertEqual(counted, "red bicycle")
        self.assertTrue(estimated)
        report = json.loads(report_text)
        self.assertEqual(report["counting_method"], LOCAL)
        self.assertFalse(report["model_tokenizer_verified"])

    def test_special_token_setting_is_applied(self):
        without_specials = self.count(add_special_tokens=False)
        with_specials = self.count(add_special_tokens=True)
        self.assertEqual(without_specials[0], 2)
        self.assertEqual(with_specials[0], 3)
        self.assertTrue(json.loads(with_specials[3])["add_special_tokens"])

    def test_saved_truncation_and_padding_do_not_distort_usage(self):
        tokenizer = tokenizers.Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer.enable_truncation(max_length=1)
        tokenizer.save(str(self.tokenizer_path))
        self.assertEqual(self.count()[0], 2)

        tokenizer.no_truncation()
        tokenizer.enable_padding(length=10, pad_id=0, pad_token="[UNK]")
        tokenizer.save(str(self.tokenizer_path))
        self.assertEqual(self.count()[0], 2)

    def test_invalid_tokenizer_file_is_explained(self):
        self.tokenizer_path.write_text('{"not":"a tokenizer"}', encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self.count()
        self.assertIn("tokenizer", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
