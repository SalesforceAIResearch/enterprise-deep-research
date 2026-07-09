import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib.util

# Load search_tools.py directly to avoid eager __init__.py imports
search_tools_path = (
    Path(__file__).resolve().parent.parent / "src" / "tools" / "search_tools.py"
)
spec = importlib.util.spec_from_file_location("search_tools", search_tools_path)
search_tools_mod = importlib.util.module_from_spec(spec)
sys.modules["search_tools"] = search_tools_mod
spec.loader.exec_module(search_tools_mod)
YouSearchTool = search_tools_mod.YouSearchTool


def test_you_search_tool_run_success():
    """should return formatted results when you_deep_search succeeds"""
    tool = YouSearchTool()

    mock_results = {
        "results": [
            {
                "title": "You.com Result",
                "url": "https://example.com/page",
                "content": "snippet",
                "raw_content": "full content here",
                "score": 0.9,
            }
        ]
    }

    with patch("src.utils_you.you_deep_search", return_value=mock_results) as mock_search:
        result = tool._run("test query", top_k=1)

    assert result["search_string"] == "test query"
    assert result["tools"] == ["you_search"]
    assert len(result["formatted_sources"]) == 1
    assert "You.com Result" in result["formatted_sources"][0]
    assert len(result["raw_contents"]) == 1
    assert result["raw_contents"][0] == "full content here"
    assert "example.com" in result["domains"]


def test_you_search_tool_run_dict_query():
    """should extract query string from dict input"""
    tool = YouSearchTool()

    mock_results = {"results": []}

    with patch("src.utils_you.you_deep_search", return_value=mock_results) as mock_search:
        result = tool._run({"query": "dict query"}, top_k=1)

    assert result["search_string"] == "dict query"
    mock_search.assert_called_once()
    call_args = mock_search.call_args
    assert call_args[1]["query"] == "dict query"


def test_you_search_tool_run_fallback_keys():
    """should fallback to other keys when 'query' is missing in dict"""
    tool = YouSearchTool()

    mock_results = {"results": []}

    with patch("src.utils_you.you_deep_search", return_value=mock_results) as mock_search:
        result = tool._run({"text": "fallback text"}, top_k=1)

    assert result["search_string"] == "fallback text"


def test_you_search_tool_run_import_error():
    """should fallback to mock search when src.utils_you is not importable"""
    tool = YouSearchTool()

    with patch(
        "src.utils_you.you_deep_search",
        side_effect=ImportError("No module named 'src.utils_you'"),
    ):
        result = tool._run("test query", top_k=1)

    assert result["search_string"] == "test query"
    assert result["tools"] == ["you_search"]
    assert "error" not in result


def test_you_search_tool_run_exception():
    """should return empty safe result on unexpected exception"""
    tool = YouSearchTool()

    with patch("src.utils_you.you_deep_search", side_effect=Exception("boom")):
        result = tool._run("test query", top_k=1)

    assert result["formatted_sources"] == []
    assert result["raw_contents"] == []
    assert result["tools"] == ["you_search"]


def test_you_search_tool_raw_content_truncation():
    """should truncate raw_content when it exceeds MAX_RAW_CONTENT_WORDS"""
    tool = YouSearchTool()

    long_content = "word " * 2500  # exceeds 2000 word limit
    mock_results = {
        "results": [
            {
                "title": "Long",
                "url": "https://example.com",
                "raw_content": long_content,
            }
        ]
    }

    with patch("src.utils_you.you_deep_search", return_value=mock_results):
        result = tool._run("test", top_k=1)

    words = result["raw_contents"][0].split()
    assert len(words) <= 2000


def test_you_search_tool_name_and_description():
    """should expose expected name and description"""
    tool = YouSearchTool()
    assert tool.name == "you_search"
    assert "You.com" in tool.description
