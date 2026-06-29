"""You.com search integration for enterprise-deep-research.

Provides a `you_deep_search` function that calls the You.com Search API
and returns results in the same format as `general_deep_search` from
`src.utils`, so it can be used as a drop-in alternative search provider.
"""

import logging
import os

import requests
from langsmith import traceable

logger = logging.getLogger("you_deep_search")

YOUCOM_SEARCH_URL = "https://api.you.com/v1/agents/search"
MAX_RESULTS = 20


@traceable
def you_deep_search(
    query: str,
    include_raw_content: bool = True,
    top_k: int = 3,
    config=None,
) -> dict:
    """General web search using You.com Search API.

    Args:
        query: The search query to execute.
        include_raw_content: Whether to include raw page content (not supported
            by You.com search API — snippets only).
        top_k: Maximum number of results to return after deduplication.
        config: RunnableConfig for LangSmith tracing (unused but kept for
            interface compatibility with general_deep_search).

    Returns:
        dict with the same shape as general_deep_search:
            - results (list): List of dicts with title, url, content, raw_content, score
            - search_string (str): The query that was searched
            - response_time (float): Approximate response time
    """
    api_key = os.getenv("YDC_API_KEY")
    if not api_key:
        raise ValueError(
            "YDC_API_KEY environment variable is not set"
        )

    query = query.strip()
    if not query:
        return {"results": [], "search_string": "", "response_time": 0}

    # Truncate long queries to a reasonable length
    if len(query) > 400:
        query = query[:397] + "..."

    try:
        response = requests.post(
            YOUCOM_SEARCH_URL,
            headers={
                "X-API-Key": api_key,
                "Content-Type": "application/json",
            },
            json={"query": query, "max_results": MAX_RESULTS},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as e:
        logger.error(f"You.com API HTTP error: {e}")
        return {
            "results": [],
            "error": f"You.com API error: {e}",
            "search_string": query,
            "response_time": 0,
        }
    except Exception as e:
        logger.error(f"You.com search error: {e}")
        return {
            "results": [],
            "error": str(e),
            "search_string": query,
            "response_time": 0,
        }

    # Map You.com results to the Tavily-compatible format
    raw_results = data.get("results", [])
    mapped_results = []
    seen_urls = set()

    for item in raw_results:
        url = item.get("url", "").strip()
        if not url or url in seen_urls:
            continue

        snippet = item.get("snippet", "") or item.get("content", "")
        mapped_results.append({
            "title": item.get("title", "Untitled"),
            "url": url,
            "content": snippet,
            "raw_content": snippet if include_raw_content else "",
            "score": item.get("score", 0.5),
        })
        seen_urls.add(url)

    # Sort by score descending and take top_k
    mapped_results = sorted(
        mapped_results, key=lambda x: x.get("score", 0), reverse=True
    )[:top_k]

    logger.info(
        f"You.com search returned {len(mapped_results)} results for query: {query}"
    )

    return {
        "results": mapped_results,
        "search_string": query,
        "response_time": data.get("response_time", 0),
    }
