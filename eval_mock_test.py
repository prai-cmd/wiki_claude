"""Mock tests for eval.py scoring functions and run_eval."""

import unittest
from unittest.mock import patch, MagicMock, call

import eval as ev


def _agent_response(answer, extracts=None, num_searches=None, tools_called=None):
    extracts = extracts or []
    return {
        "answer": answer,
        "retrieved_extracts": extracts,
        "num_searches": num_searches if num_searches is not None else len(extracts),
        "tools_called": tools_called or [],
    }


# ---------------------------------------------------------------------------
# _parse_counts — no mocking needed, pure function
# ---------------------------------------------------------------------------

class TestParseCounts(unittest.TestCase):
    def test_parses_both_keys(self):
        text = "contradicted: 2\ntotal: 10"
        a, b = ev._parse_counts(text, "contradicted", "total")
        self.assertEqual(a, 2)
        self.assertEqual(b, 10)

    def test_parses_grounded_total(self):
        a, b = ev._parse_counts("grounded: 9\ntotal: 10", "grounded", "total")
        self.assertEqual(a, 9)
        self.assertEqual(b, 10)

    def test_case_insensitive(self):
        a, b = ev._parse_counts("Contradicted: 3\nTotal: 15", "contradicted", "total")
        self.assertEqual(a, 3)
        self.assertEqual(b, 15)

    def test_missing_key_a_returns_zero(self):
        a, b = ev._parse_counts("total: 8", "contradicted", "total")
        self.assertEqual(a, 0)
        self.assertEqual(b, 8)

    def test_missing_key_b_returns_default_one(self):
        a, b = ev._parse_counts("contradicted: 1", "contradicted", "total")
        self.assertEqual(a, 1)
        self.assertEqual(b, 1)

    def test_extra_whitespace_handled(self):
        a, b = ev._parse_counts("contradicted:  0\ntotal:  5", "contradicted", "total")
        self.assertEqual(a, 0)
        self.assertEqual(b, 5)

    def test_empty_string_returns_defaults(self):
        a, b = ev._parse_counts("", "contradicted", "total")
        self.assertEqual(a, 0)
        self.assertEqual(b, 1)

    def test_ignores_unrelated_lines(self):
        text = "Some preamble\ncontradicted: 4\nsome other line\ntotal: 20"
        a, b = ev._parse_counts(text, "contradicted", "total")
        self.assertEqual(a, 4)
        self.assertEqual(b, 20)


# ---------------------------------------------------------------------------
# score_answer_correctness
# ---------------------------------------------------------------------------

class TestScoreAnswerCorrectness(unittest.TestCase):
    @patch("eval._judge", return_value="4.0")
    def test_returns_correct_float(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 4.0)

    @patch("eval._judge", return_value="3.5")
    def test_parses_decimal_score(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 3.5)

    @patch("eval._judge", return_value="5.0")
    def test_returns_maximum(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 5.0)

    @patch("eval._judge", return_value="1.0")
    def test_returns_minimum(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", return_value="6.0")
    def test_clamps_above_5(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 5.0)

    @patch("eval._judge", return_value="0.5")
    def test_clamps_below_1(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", return_value="not a number")
    def test_returns_1_on_parse_failure(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", side_effect=Exception("api error"))
    def test_returns_1_on_exception(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", return_value="4.0")
    def test_judge_receives_question_answer_expected(self, mock_judge):
        ev.score_answer_correctness("My question", "My answer", "My expected")
        prompt = mock_judge.call_args.args[0]
        self.assertIn("My question", prompt)
        self.assertIn("My answer", prompt)
        self.assertIn("My expected", prompt)

    @patch("eval._judge", return_value="4.0")
    def test_return_type_is_float(self, _):
        score = ev.score_answer_correctness("Q", "A", "E")
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# score_hallucination
# ---------------------------------------------------------------------------

class TestScoreHallucination(unittest.TestCase):
    def test_returns_zero_when_extracts_empty(self):
        score = ev.score_hallucination("Some answer.", [])
        self.assertEqual(score, 0.0)

    @patch("eval._judge", return_value="contradicted: 0\ntotal: 5")
    def test_zero_contradictions(self, _):
        score = ev.score_hallucination("Answer.", ["Extract text."])
        self.assertAlmostEqual(score, 0.0)

    @patch("eval._judge", return_value="contradicted: 2\ntotal: 10")
    def test_correct_fraction(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertAlmostEqual(score, 0.2)

    @patch("eval._judge", return_value="contradicted: 10\ntotal: 10")
    def test_all_contradicted(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", return_value="contradicted: 0\ntotal: 0")
    def test_zero_total_returns_zero(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertEqual(score, 0.0)

    @patch("eval._judge", side_effect=Exception("api error"))
    def test_returns_zero_on_exception(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertEqual(score, 0.0)

    @patch("eval._judge", return_value="contradicted: 1\ntotal: 4")
    def test_multiple_extracts_joined(self, mock_judge):
        ev.score_hallucination("Answer.", ["Extract A.", "Extract B."])
        prompt = mock_judge.call_args.args[0]
        self.assertIn("Extract A.", prompt)
        self.assertIn("Extract B.", prompt)

    @patch("eval._judge", return_value="contradicted: 1\ntotal: 10")
    def test_result_clamped_to_one(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    @patch("eval._judge", return_value="contradicted: 1\ntotal: 10")
    def test_return_type_is_float(self, _):
        score = ev.score_hallucination("Answer.", ["Extract."])
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# score_grounding
# ---------------------------------------------------------------------------

class TestScoreGrounding(unittest.TestCase):
    def test_returns_zero_when_extracts_empty(self):
        score = ev.score_grounding("Answer.", [], "strict")
        self.assertEqual(score, 0.0)

    @patch("eval._judge", return_value="grounded: 9\ntotal: 10")
    def test_correct_fraction_strict(self, _):
        score = ev.score_grounding("Answer.", ["Extract."], "strict")
        self.assertAlmostEqual(score, 0.9)

    @patch("eval._judge", return_value="grounded: 10\ntotal: 10")
    def test_correct_fraction_lenient(self, _):
        score = ev.score_grounding("Answer.", ["Extract."], "lenient")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", return_value="grounded: 0\ntotal: 0")
    def test_zero_total_returns_one(self, _):
        score = ev.score_grounding("Answer.", ["Extract."], "strict")
        self.assertAlmostEqual(score, 1.0)

    @patch("eval._judge", side_effect=Exception("api error"))
    def test_returns_zero_on_exception(self, _):
        score = ev.score_grounding("Answer.", ["Extract."], "strict")
        self.assertEqual(score, 0.0)

    @patch("eval._judge", return_value="grounded: 5\ntotal: 5")
    def test_strict_keyword_in_prompt(self, mock_judge):
        ev.score_grounding("Answer.", ["Extract."], "strict")
        prompt = mock_judge.call_args.args[0]
        self.assertIn("strict", prompt.lower())

    @patch("eval._judge", return_value="grounded: 5\ntotal: 5")
    def test_lenient_keyword_in_prompt(self, mock_judge):
        ev.score_grounding("Answer.", ["Extract."], "lenient")
        prompt = mock_judge.call_args.args[0]
        self.assertIn("lenient", prompt.lower())

    @patch("eval._judge", return_value="grounded: 5\ntotal: 5")
    def test_multiple_extracts_joined(self, mock_judge):
        ev.score_grounding("Answer.", ["Extract A.", "Extract B."], "lenient")
        prompt = mock_judge.call_args.args[0]
        self.assertIn("Extract A.", prompt)
        self.assertIn("Extract B.", prompt)

    @patch("eval._judge", return_value="grounded: 3\ntotal: 10")
    def test_return_type_is_float(self, _):
        score = ev.score_grounding("Answer.", ["Extract."], "strict")
        self.assertIsInstance(score, float)


# ---------------------------------------------------------------------------
# score_refusal
# ---------------------------------------------------------------------------

class TestScoreRefusal(unittest.TestCase):
    @patch("eval._judge", return_value="yes")
    def test_returns_one_on_yes(self, _):
        self.assertEqual(ev.score_refusal("I cannot answer that."), 1.0)

    @patch("eval._judge", return_value="no")
    def test_returns_zero_on_no(self, _):
        self.assertEqual(ev.score_refusal("The answer is Paris."), 0.0)

    @patch("eval._judge", return_value="YES")
    def test_case_insensitive_yes(self, _):
        self.assertEqual(ev.score_refusal("I cannot answer."), 1.0)

    @patch("eval._judge", return_value="no, it answered")
    def test_starts_with_no_returns_zero(self, _):
        self.assertEqual(ev.score_refusal("Some answer."), 0.0)

    @patch("eval._judge", side_effect=Exception("api error"))
    def test_returns_zero_on_exception(self, _):
        self.assertEqual(ev.score_refusal("Answer."), 0.0)

    @patch("eval._judge", return_value="yes")
    def test_answer_included_in_prompt(self, mock_judge):
        ev.score_refusal("I cannot answer that question.")
        prompt = mock_judge.call_args.args[0]
        self.assertIn("I cannot answer that question.", prompt)

    @patch("eval._judge", return_value="yes")
    def test_return_type_is_float(self, _):
        self.assertIsInstance(ev.score_refusal("I cannot answer."), float)


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

class TestRunEval(unittest.TestCase):
    def _case(self, category, grounding_mode="strict", requires_tool=True):
        return {
            "question": f"Question about {category}",
            "category": category,
            "expected_answer": "Expected answer.",
            "grounding_mode": grounding_mode,
            "requires_tool": requires_tool,
        }

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.02)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_factual_case_uses_all_three_scorers(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Good answer.", ["Extract."])
        result = ev.run_eval([self._case("factual")])
        mock_correct.assert_called_once()
        mock_hall.assert_called_once()
        mock_ground.assert_called_once()

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.02)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_comparative_case_uses_all_three_scorers(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Good answer.", ["Extract."])
        result = ev.run_eval([self._case("comparative", "lenient")])
        mock_correct.assert_called_once()
        mock_hall.assert_called_once()
        mock_ground.assert_called_once()

    @patch("eval.score_refusal", return_value=1.0)
    @patch("eval.score_grounding")
    @patch("eval.score_hallucination")
    @patch("eval.score_answer_correctness")
    @patch("agent.ask_agent")
    def test_out_of_scope_uses_only_refusal(
        self, mock_agent, mock_correct, mock_hall, mock_ground, mock_refusal
    ):
        mock_agent.return_value = _agent_response("I cannot answer that.", [])
        ev.run_eval([self._case("out_of_scope", requires_tool=False)])
        mock_refusal.assert_called_once()
        mock_correct.assert_not_called()
        mock_hall.assert_not_called()
        mock_ground.assert_not_called()

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination")
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_coverage_edge_uses_correctness_and_grounding_only(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        ev.run_eval([self._case("coverage_edge", "lenient")])
        mock_correct.assert_called_once()
        mock_ground.assert_called_once()
        mock_hall.assert_not_called()

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.0)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_rate_limit_failure_excluded_from_metrics(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("RATE_LIMIT_FAILURE: timeout", [])
        result = ev.run_eval([self._case("factual")])
        self.assertEqual(result["excluded_rate_limit"], 1)
        self.assertEqual(result["total_evaluated"], 0)
        mock_correct.assert_not_called()
        mock_hall.assert_not_called()
        mock_ground.assert_not_called()

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.0)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_excluded_count_increments_per_failure(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("RATE_LIMIT_FAILURE: timeout", [])
        result = ev.run_eval([self._case("factual"), self._case("factual")])
        self.assertEqual(result["excluded_rate_limit"], 2)

    @patch("eval.score_grounding", return_value=0.95)
    @patch("eval.score_hallucination", return_value=0.02)
    @patch("eval.score_answer_correctness", return_value=4.5)
    @patch("agent.ask_agent")
    def test_return_dict_has_all_required_keys(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        result = ev.run_eval([self._case("factual")])
        for key in (
            "total_cases", "total_evaluated", "excluded_rate_limit",
            "avg_answer_correctness", "avg_hallucination_rate",
            "avg_grounding", "avg_refusal_rate", "per_case",
        ):
            self.assertIn(key, result)

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.0)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_total_cases_and_evaluated_counts(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        result = ev.run_eval([self._case("factual"), self._case("factual")])
        self.assertEqual(result["total_cases"], 2)
        self.assertEqual(result["total_evaluated"], 2)

    @patch("eval.score_grounding", return_value=0.8)
    @patch("eval.score_hallucination", return_value=0.1)
    @patch("eval.score_answer_correctness", return_value=3.0)
    @patch("agent.ask_agent")
    def test_averages_computed_correctly(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        result = ev.run_eval([self._case("factual"), self._case("factual")])
        self.assertAlmostEqual(result["avg_answer_correctness"], 3.0)
        self.assertAlmostEqual(result["avg_hallucination_rate"], 0.1)
        self.assertAlmostEqual(result["avg_grounding"], 0.8)

    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.0)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_per_case_list_populated(
        self, mock_agent, mock_correct, mock_hall, mock_ground
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        result = ev.run_eval([self._case("factual")])
        self.assertEqual(len(result["per_case"]), 1)
        self.assertIn("answer", result["per_case"][0])
        self.assertIn("category", result["per_case"][0])

    @patch("eval.score_refusal", return_value=1.0)
    @patch("eval.score_grounding", return_value=0.9)
    @patch("eval.score_hallucination", return_value=0.0)
    @patch("eval.score_answer_correctness", return_value=4.0)
    @patch("agent.ask_agent")
    def test_mixed_categories_route_correctly(
        self, mock_agent, mock_correct, mock_hall, mock_ground, mock_refusal
    ):
        mock_agent.return_value = _agent_response("Answer.", ["Extract."])
        cases = [self._case("factual"), self._case("out_of_scope", requires_tool=False)]
        ev.run_eval(cases)
        # factual triggers correctness; out_of_scope triggers refusal
        self.assertEqual(mock_correct.call_count, 1)
        self.assertEqual(mock_refusal.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
