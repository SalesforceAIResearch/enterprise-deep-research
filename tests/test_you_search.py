import os
from unittest.mock import patch, MagicMock
import pytest
from src.utils_you import you_deep_search


def test_you_deep_search_missing_api_key(monkeypatch):
    """should raise ValueError when YDC_API_KEY is not set"""
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="YDC_API_KEY"):
        you_deep_search("test query")


def test_you_deep_search_empty_query_returns_empty(monkeypatch):
    """should return empty results for empty query"""
    monkeypatch.setenv("YDC_API_KEY", "test_key")
    result = you_deep_search("   ")
    assert result["results"] == []
    assert result["search_string"] == ""


def test_you_deep_search_basic_success(monkeypatch):
    """should return formatted results on successful API call"""
    monkeypatch.setenv("YDC_API_KEY", "test_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com/article",
                "snippet": "A short summary.",
                "markdown": "Full markdown content.",
                "score": 0.95,
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch("src.utils_you.requests.post", return_value=mock_response):
        result = you_deep_search("test query", include_raw_content=True, top_k=1)

    assert len(result["results"]) == 1
    r = result["results"][0]
    assert r["title"] == "Test Title"
    assert r["url"] == "https://example.com/article"
    assert r["content"] == "A short summary."
    assert r["raw_content"] == "Full markdown content."
    assert r["score"] == 0.95


def test_you_deep_search_no_raw_content(monkeypatch):
    """should set raw_content to None when include_raw_content=False"""
    monkeypatch.setenv("YDC_API_KEY", "test_key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Test Title",
                "url": "https://example.com/article",
                "snippet": "A short summary.",
                "markdown": "Full markdown content.",
            }
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch("src.utils_you.requests.post", return_value=mock_response):
        result = you_deep_search("test query", include_raw_content=False)

    assert result["results"][0]["raw_content"] is None


def test_you_deep_search_401_error(monkeypatch):
    """should return empty results with error on 401"""
    monkeypatch.setenv("YDC_API_KEY", "bad_key")

    error_response = MagicMock()
    error_response.status_code = 401
    error_response.text = "Unauthorized"
    from requests.exceptions import HTTPError
    error_response.raise_for_status.side_effect = HTTPError(response=error_response)

    with patch("src.utils_you.requests.post", return_value=error_response):
        result = you_deep_search("test query")

    assert result["results"] == []
    assert "error" in result


def test_you_deep_search_network_error(monkeypatch):
    """should return empty results with error message on unexpected failure"""
    monkeypatch.setenv("YDC_API_KEY", "test_key")

    with patch("src.utils_you.requests.post", side_effect=Exception("Connection refused")):
        result = you_deep_search("test query")

    assert result["results"] == []
    assert "error" in result
