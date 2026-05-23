"""Handles all Wikipedia API calls via the MediaWiki REST API."""

_cache = {}


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
    pass
