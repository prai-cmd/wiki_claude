"""Mock tests for wikipedia.py."""

import unittest
from unittest.mock import MagicMock, call, patch

import wikipedia


def _make_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _search_resp(titles):
    return _make_response({"pages": [{"title": t} for t in titles]})


def _summary_resp(title, extract, url="https://en.wikipedia.org/wiki/Test"):
    return _make_response({
        "title": title,
        "extract": extract,
        "content_urls": {"desktop": {"page": url}},
    })


def _long_extract(word_count=600):
    return " ".join(["word"] * word_count)


# ---------------------------------------------------------------------------
# Helpers — clear cache before every test
# ---------------------------------------------------------------------------

class WikipediaTestCase(unittest.TestCase):
    def setUp(self):
        wikipedia._cache.clear()


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_returns_formatted_string(self, mock_get, mock_sleep):
        extract = "A" * 200
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", extract,
                          "https://en.wikipedia.org/wiki/Python"),
        ]
        result = wikipedia.search_wikipedia("Python")
        self.assertTrue(result.startswith("Wikipedia: Python (programming language)"))
        self.assertIn(extract, result)
        self.assertIn("Source: https://en.wikipedia.org/wiki/Python", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_return_format_exact_structure(self, mock_get, mock_sleep):
        extract = "B" * 150
        mock_get.side_effect = [
            _search_resp(["Marie Curie"]),
            _summary_resp("Marie Curie", extract, "https://en.wikipedia.org/wiki/Marie_Curie"),
        ]
        result = wikipedia.search_wikipedia("Marie Curie")
        expected = f"Wikipedia: Marie Curie\n\n{extract}\n\nSource: https://en.wikipedia.org/wiki/Marie_Curie"
        self.assertEqual(result, expected)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_makes_two_requests(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "C" * 200),
        ]
        wikipedia.search_wikipedia("Python")
        self.assertEqual(mock_get.call_count, 2)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_user_agent_header_on_search(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "D" * 200),
        ]
        wikipedia.search_wikipedia("Python")
        first_call_kwargs = mock_get.call_args_list[0][1]
        self.assertIn("User-Agent", first_call_kwargs["headers"])

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_user_agent_header_on_summary(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "E" * 200),
        ]
        wikipedia.search_wikipedia("Python")
        second_call_kwargs = mock_get.call_args_list[1][1]
        self.assertIn("User-Agent", second_call_kwargs["headers"])


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_result_stored_in_cache(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "F" * 200),
        ]
        result = wikipedia.search_wikipedia("Python")
        self.assertIn("Python", wikipedia._cache)
        self.assertEqual(wikipedia._cache["Python"], result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_second_call_uses_cache(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "G" * 200),
        ]
        first = wikipedia.search_wikipedia("Python")
        second = wikipedia.search_wikipedia("Python")
        self.assertEqual(first, second)
        self.assertEqual(mock_get.call_count, 2)  # no extra calls on cache hit

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_rate_limit_failure_not_cached(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("timeout")
        wikipedia.search_wikipedia("Python")
        self.assertNotIn("Python", wikipedia._cache)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_no_results_not_cached(self, mock_get, mock_sleep):
        mock_get.return_value = _make_response({"pages": []})
        wikipedia.search_wikipedia("xyzzy_nonexistent")
        self.assertNotIn("xyzzy_nonexistent", wikipedia._cache)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_empty_pages_returns_no_article_found(self, mock_get, mock_sleep):
        mock_get.return_value = _make_response({"pages": []})
        result = wikipedia.search_wikipedia("xyzzy_nonexistent")
        self.assertEqual(result, "No Wikipedia article found for: xyzzy_nonexistent")

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_stub_extract_under_100_chars_returns_no_article_found(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Stub Article"]),
            _summary_resp("Stub Article", "Too short."),
        ]
        result = wikipedia.search_wikipedia("stub query")
        self.assertEqual(result, "No Wikipedia article found for: stub query")

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_extract_exactly_99_chars_is_a_miss(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Short Article"]),
            _summary_resp("Short Article", "x" * 99),
        ]
        result = wikipedia.search_wikipedia("short query")
        self.assertEqual(result, "No Wikipedia article found for: short query")

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_extract_exactly_100_chars_is_a_hit(self, mock_get, mock_sleep):
        extract = "x" * 100
        mock_get.side_effect = [
            _search_resp(["Borderline Article"]),
            _summary_resp("Borderline Article", extract),
        ]
        result = wikipedia.search_wikipedia("borderline query")
        self.assertIn("Wikipedia: Borderline Article", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_network_failure_returns_rate_limit_failure(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("connection refused")
        result = wikipedia.search_wikipedia("Python")
        self.assertTrue(result.startswith("RATE_LIMIT_FAILURE:"))
        self.assertIn("connection refused", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_summary_network_failure_returns_rate_limit_failure(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            Exception("summary timeout"),
            Exception("summary timeout"),
            Exception("summary timeout"),
        ]
        result = wikipedia.search_wikipedia("Python")
        self.assertTrue(result.startswith("RATE_LIMIT_FAILURE:"))


# ---------------------------------------------------------------------------
# Retry / backoff
# ---------------------------------------------------------------------------

class TestRetryBehaviour(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_retries_three_times_on_failure(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("timeout")
        wikipedia.search_wikipedia("Python")
        self.assertEqual(mock_get.call_count, 3)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_backoff_waits_are_2_4_6(self, mock_get, mock_sleep):
        mock_get.side_effect = Exception("timeout")
        wikipedia.search_wikipedia("Python")
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertEqual(sleep_calls[:3], [2, 4, 6])

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_succeeds_on_third_attempt(self, mock_get, mock_sleep):
        extract = "H" * 200
        mock_get.side_effect = [
            Exception("fail 1"),
            Exception("fail 2"),
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", extract),
        ]
        result = wikipedia.search_wikipedia("Python")
        self.assertIn("Wikipedia: Python", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_no_sleep_on_first_success(self, mock_get, mock_sleep):
        mock_get.side_effect = [
            _search_resp(["Python (programming language)"]),
            _summary_resp("Python (programming language)", "I" * 200),
        ]
        wikipedia.search_wikipedia("Python")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Disambiguation handling
# ---------------------------------------------------------------------------

class TestDisambiguation(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_skips_disambiguation_page(self, mock_get, mock_sleep):
        extract = "J" * 200
        mock_get.side_effect = [
            _search_resp(["Mercury (disambiguation)", "Mercury (planet)"]),
            _summary_resp("Mercury (planet)", extract,
                          "https://en.wikipedia.org/wiki/Mercury_(planet)"),
        ]
        result = wikipedia.search_wikipedia("Mercury")
        self.assertIn("Mercury (planet)", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_falls_back_to_disambiguation_if_all_are_disambiguation(self, mock_get, mock_sleep):
        extract = "K" * 200
        mock_get.side_effect = [
            _search_resp(["Mercury (disambiguation)"]),
            _summary_resp("Mercury (disambiguation)", extract),
        ]
        result = wikipedia.search_wikipedia("Mercury")
        self.assertIn("Wikipedia:", result)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_non_disambiguation_first_result_used_directly(self, mock_get, mock_sleep):
        extract = "L" * 200
        mock_get.side_effect = [
            _search_resp(["Python (programming language)", "Python (disambiguation)"]),
            _summary_resp("Python (programming language)", extract),
        ]
        result = wikipedia.search_wikipedia("Python")
        self.assertIn("Python (programming language)", result)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

class TestTruncation(WikipediaTestCase):
    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_extract_over_500_words_is_truncated(self, mock_get, mock_sleep):
        long_extract = _long_extract(600)
        mock_get.side_effect = [
            _search_resp(["Long Article"]),
            _summary_resp("Long Article", long_extract),
        ]
        result = wikipedia.search_wikipedia("long query")
        extract_part = result.split("\n\n")[1]
        self.assertEqual(len(extract_part.split()), 500)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_extract_under_500_words_not_truncated(self, mock_get, mock_sleep):
        short_extract = _long_extract(300)
        mock_get.side_effect = [
            _search_resp(["Short Article"]),
            _summary_resp("Short Article", short_extract),
        ]
        result = wikipedia.search_wikipedia("short query")
        extract_part = result.split("\n\n")[1]
        self.assertEqual(len(extract_part.split()), 300)

    @patch("wikipedia.time.sleep")
    @patch("wikipedia.requests.get")
    def test_extract_exactly_500_words_not_truncated(self, mock_get, mock_sleep):
        exact_extract = _long_extract(500)
        mock_get.side_effect = [
            _search_resp(["Exact Article"]),
            _summary_resp("Exact Article", exact_extract),
        ]
        result = wikipedia.search_wikipedia("exact query")
        extract_part = result.split("\n\n")[1]
        self.assertEqual(len(extract_part.split()), 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
