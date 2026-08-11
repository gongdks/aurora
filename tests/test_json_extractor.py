"""Tests for agent.utils.json_extractor — JSON extraction from LLM responses."""

import pytest
from agent.utils.json_extractor import extract_json, extract_json_safe


class TestExtractJson:
    """Tests for extract_json()."""

    def test_markdown_json_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_markdown_block_without_json_tag(self):
        text = '```\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_raw_json_object(self):
        text = 'Some text {"name": "test", "count": 42} more text'
        result = extract_json(text)
        assert result == {"name": "test", "count": 42}

    def test_raw_json_array(self):
        text = "Here is a list: [1, 2, 3] end"
        result = extract_json(text)
        assert result == [1, 2, 3]

    def test_nested_json_object(self):
        text = '```json\n{"outer": {"inner": [1, 2, 3]}, "flag": true}\n```'
        result = extract_json(text)
        assert result == {"outer": {"inner": [1, 2, 3]}, "flag": True}

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="响应为空"):
            extract_json("")

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="无法从响应中提取 JSON"):
            extract_json("Just plain text without any JSON at all")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="响应为空"):
            extract_json("   \n\t  ")


class TestExtractJsonSafe:
    """Tests for extract_json_safe()."""

    def test_valid_json(self):
        result = extract_json_safe('{"a": 1}')
        assert result == {"a": 1}

    def test_invalid_returns_default_dict(self):
        # extract_json_safe returns {} by default when parsing fails
        # (not None — that's the no-default behavior)
        result = extract_json_safe("not json")
        assert result == {}

    def test_default_none_still_returns_empty_dict(self):
        # None means "no default was explicitly provided" — returns {}
        result = extract_json_safe("not json", default=None)
        assert result == {}

    def test_custom_default(self):
        result = extract_json_safe("not json", default={"fallback": True})
        assert result == {"fallback": True}

    def test_no_default_returns_empty_dict(self):
        result = extract_json_safe("not json")
        assert result == {}

    def test_array_result_wraps_in_dict(self):
        result = extract_json_safe("[1, 2, 3]")
        assert result == {"result": [1, 2, 3]}
