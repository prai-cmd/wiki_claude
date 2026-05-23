"""Handles all Wikipedia API calls via the MediaWiki REST API."""

import time

import requests

_cache = {}

_HEADERS = {"User-Agent": "wiki-claude-qa/1.0 (https://github.com/prai-cmd/wiki_claude)"}
_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_BACKOFF = [2, 4, 6]


def _get(url, **kwargs):
    """GET with up to 3 retries and exponential backoff. Returns Response or raises."""
    last_exc = None
    for wait in _BACKOFF:
        try:
            response = requests.get(url, headers=_HEADERS, timeout=10, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            time.sleep(wait)
    raise last_exc


def search_wikipedia(query: str) -> str:
    """Search Wikipedia and return a formatted article summary.

    Retrieval flow:
        1. Check _cache first using query as the key. If a cached result
           exists, return it immediately without making any network calls.
        2. Call the MediaWiki REST API search endpoint with the query to
           obtain a ranked list of article titles. Take the top result's
           title and proceed to fetch. If the top result appears to be a
           disambiguation page, fall back to the second-ranked result.
        3. Call the MediaWiki REST API page summary endpoint for the
           resolved title to fetch the article's title, plain-text extract,
           and canonical URL.

    Retry behaviour:
        Each network call is retried up to 3 attempts with exponential
        backoff: wait 2 s after the first failure, 4 s after the second,
        6 s after the third. If all 3 attempts fail due to rate limiting
        or a persistent network error, return the string
        "RATE_LIMIT_FAILURE: <reason>" and do not cache the result.

    Post-processing:
        Truncate the extract to approximately 500 words before storing
        or returning it.

    Caching:
        Store the formatted result string in _cache under the original
        query key before returning, so subsequent identical queries are
        served from cache.

    Return format:
        "Wikipedia: {title}\\n\\n{extract}\\n\\nSource: {url}"

    Error cases:
        - No results: the search endpoint returns an empty list.
          Return "No Wikipedia article found for: {query}".
        - Network failure: a request raises an exception on all retries.
          Return "RATE_LIMIT_FAILURE: {exception message}".
        - Stub article: the fetched extract is under 100 characters,
          treated as a content miss. Return
          "No Wikipedia article found for: {query}".
    """
    if query in _cache:
        return _cache[query]

    # --- Search phase ---
    try:
        search_resp = _get(_SEARCH_URL, params={"q": query, "limit": 5})
    except Exception as exc:
        return f"RATE_LIMIT_FAILURE: {exc}"

    pages = search_resp.json().get("pages", [])
    if not pages:
        return f"No Wikipedia article found for: {query}"

    # Resolve disambiguation: skip the first result if it is a disambiguation page
    title = None
    for page in pages:
        candidate = page.get("title", "")
        if "(disambiguation)" not in candidate.lower():
            title = candidate
            break
    if title is None:
        title = pages[0].get("title", "")

    # --- Summary phase ---
    try:
        summary_resp = _get(_SUMMARY_URL.format(title=requests.utils.quote(title, safe="")))
    except Exception as exc:
        return f"RATE_LIMIT_FAILURE: {exc}"

    data = summary_resp.json()
    extract = data.get("extract", "")
    url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    article_title = data.get("title", title)

    if len(extract) < 100:
        return f"No Wikipedia article found for: {query}"

    # Truncate to ~500 words
    words = extract.split()
    if len(words) > 500:
        extract = " ".join(words[:500])

    result = f"Wikipedia: {article_title}\n\n{extract}\n\nSource: {url}"
    _cache[query] = result
    return result
