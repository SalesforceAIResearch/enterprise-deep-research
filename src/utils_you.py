import os
import requests
import logging
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger("you_search")

@traceable
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(requests.exceptions.HTTPError),
)
def you_deep_search(query, include_raw_content=True, top_k=3, config=None):
    """Web search using You.com API"""
    api_key = os.getenv("YDC_API_KEY")
    if not api_key:
        raise ValueError("YDC_API_KEY environment variable is not set")

    query = query.strip()
    if not query:
        return {"results": [], "search_string": "", "response_time": 0}

    url = "https://api.you.com/v1/agents/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "livecrawl": "web" if include_raw_content else "none",
        "top_k": top_k
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Map You.com response to EDR format
        results = []
        you_results = data.get("results", [])
        
        for res in you_results:
            results.append({
                "title": res.get("title", "Untitled"),
                "url": res.get("url", "No URL"),
                "content": res.get("snippet", ""),
                "raw_content": res.get("markdown", "") if include_raw_content else None,
                "score": res.get("score", 1.0)
            })
            
        return {"results": results}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return {"results": [], "error": "Invalid API key", "search_string": query}
        raise
    except Exception as e:
        logger.error(f"Unexpected error in You.com search: {str(e)}")
        return {"results": [], "error": str(e), "search_string": query}
