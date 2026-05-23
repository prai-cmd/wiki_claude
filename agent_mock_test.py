"""Mock tests for agent.py, wikipedia.py, and eval.py."""

import inspect
import unittest
from unittest.mock import MagicMock, patch

import agent
import eval as eval_module
import wikipedia


# ---------------------------------------------------------------------------
# agent.py — constants
# ---------------------------------------------------------------------------

class TestAgentConstants(unittest.TestCase):
    def test_system_prompt_is_non_empty_string(self):
        self.assertIsInstance(agent.SYSTEM_PROMPT, str)
        self.assertGreater(len(agent.SYSTEM_PROMPT), 0)

    def test_system_prompt_has_section_1(self):
        self.assertIn("Section 1", agent.SYSTEM_PROMPT)

    def test_system_prompt_has_section_2(self):
        self.assertIn("Section 2", agent.SYSTEM_PROMPT)

    def test_tools_is_list(self):
        self.assertIsInstance(agent.TOOLS, list)

    def test_tools_has_one_entry(self):
        self.assertEqual(len(agent.TOOLS), 1)

    def test_tool_name_is_search_wikipedia(self):
        self.assertEqual(agent.TOOLS[0]["name"], "search_wikipedia")

    def test_tool_has_description(self):
        self.assertIn("description", agent.TOOLS[0])
        self.assertGreater(len(agent.TOOLS[0]["description"]), 0)

    def test_tool_has_input_schema_with_query(self):
        schema = agent.TOOLS[0]["input_schema"]
        self.assertIn("properties", schema)
        self.assertIn("query", schema["properties"])

    def test_tool_query_is_required(self):
        self.assertIn("query", agent.TOOLS[0]["input_schema"]["required"])


# ---------------------------------------------------------------------------
# agent.py — ask_agent: no tool calls
# ---------------------------------------------------------------------------

class TestAskAgentNoToolCalls(unittest.TestCase):
    def _make_text_response(self, text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [block]
        return response

    @patch("agent.anthropic.Anthropic")
    def test_answer_returned_correctly(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = (
            self._make_text_response("The answer is 42.")
        )
        result = agent.ask_agent("What is the answer?")
        self.assertEqual(result["answer"], "The answer is 42.")

    @patch("agent.anthropic.Anthropic")
    def test_no_tool_calls_empty_collections(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = (
            self._make_text_response("No search needed.")
        )
        result = agent.ask_agent("What is a prime number?")
        self.assertEqual(result["tools_called"], [])
        self.assertEqual(result["num_searches"], 0)
        self.assertEqual(result["retrieved_extracts"], [])

    @patch("agent.anthropic.Anthropic")
    def test_return_dict_has_all_required_keys(self, mock_cls):
        mock_cls.return_value.messages.create.return_value = (
            self._make_text_response("Answer.")
        )
        result = agent.ask_agent("Any question?")
        for key in ("answer", "tools_called", "num_searches", "retrieved_extracts"):
            self.assertIn(key, result)

    @patch("agent.anthropic.Anthropic")
    def test_api_called_with_correct_model(self, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.messages.create.return_value = (
            self._make_text_response("Answer.")
        )
        agent.ask_agent("Test question")
        kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "claude-sonnet-4-6")

    @patch("agent.anthropic.Anthropic")
    def test_api_called_with_system_prompt(self, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.messages.create.return_value = (
            self._make_text_response("Answer.")
        )
        agent.ask_agent("Test question")
        kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"], agent.SYSTEM_PROMPT)

    @patch("agent.anthropic.Anthropic")
    def test_api_called_with_tools(self, mock_cls):
        mock_client = mock_cls.return_value
        mock_client.messages.create.return_value = (
            self._make_text_response("Answer.")
        )
        agent.ask_agent("Test question")
        kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["tools"], agent.TOOLS)


# ---------------------------------------------------------------------------
# agent.py — ask_agent: tool calls
# ---------------------------------------------------------------------------

class TestAskAgentWithToolCalls(unittest.TestCase):
    def _make_tool_response(self, tool_blocks):
        response = MagicMock()
        response.stop_reason = "tool_use"
        response.content = tool_blocks
        return response

    def _make_end_response(self, text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.stop_reason = "end_turn"
        response.content = [block]
        return response

    def _make_tool_block(self, tool_id, query):
        block = MagicMock()
        block.type = "tool_use"
        block.name = "search_wikipedia"
        block.id = tool_id
        block.input = {"query": query}
        return block

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_single_tool_call_increments_num_searches(self, mock_cls, mock_search):
        mock_search.return_value = (
            "Wikipedia: Python\n\nPython is a language.\n\nSource: https://en.wikipedia.org/wiki/Python"
        )
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([self._make_tool_block("t1", "Python programming language")]),
            self._make_end_response("Python is a high-level language."),
        ]
        result = agent.ask_agent("What is Python?")
        self.assertEqual(result["num_searches"], 1)

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_single_tool_call_records_query(self, mock_cls, mock_search):
        mock_search.return_value = "Wikipedia: Python\n\nExtract.\n\nSource: https://..."
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([self._make_tool_block("t1", "Python programming language")]),
            self._make_end_response("Python is a language."),
        ]
        result = agent.ask_agent("What is Python?")
        self.assertEqual(result["tools_called"], ["Python programming language"])

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_single_tool_call_records_extract(self, mock_cls, mock_search):
        extract = "Wikipedia: Python\n\nPython is a language.\n\nSource: https://..."
        mock_search.return_value = extract
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([self._make_tool_block("t1", "Python programming language")]),
            self._make_end_response("Python is a language."),
        ]
        result = agent.ask_agent("What is Python?")
        self.assertEqual(result["retrieved_extracts"], [extract])

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_two_tool_calls_in_one_turn(self, mock_cls, mock_search):
        mock_search.side_effect = [
            "Wikipedia: Marie Curie\n\nMarie Curie was a physicist.\n\nSource: https://...",
            "Wikipedia: Radioactivity\n\nRadioactivity is a nuclear process.\n\nSource: https://...",
        ]
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([
                self._make_tool_block("t1", "Marie Curie"),
                self._make_tool_block("t2", "Radioactivity"),
            ]),
            self._make_end_response("Marie Curie discovered radioactivity."),
        ]
        result = agent.ask_agent("What did Marie Curie discover?")
        self.assertEqual(result["num_searches"], 2)
        self.assertEqual(result["tools_called"], ["Marie Curie", "Radioactivity"])
        self.assertEqual(len(result["retrieved_extracts"]), 2)

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_extracts_preserved_in_order(self, mock_cls, mock_search):
        extract_1 = "Wikipedia: A\n\nFirst extract.\n\nSource: https://..."
        extract_2 = "Wikipedia: B\n\nSecond extract.\n\nSource: https://..."
        mock_search.side_effect = [extract_1, extract_2]
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([
                self._make_tool_block("t1", "Topic A"),
                self._make_tool_block("t2", "Topic B"),
            ]),
            self._make_end_response("Combined answer."),
        ]
        result = agent.ask_agent("Compare A and B.")
        self.assertEqual(result["retrieved_extracts"][0], extract_1)
        self.assertEqual(result["retrieved_extracts"][1], extract_2)

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_tool_calls_across_two_turns(self, mock_cls, mock_search):
        mock_search.side_effect = [
            "Wikipedia: X\n\nFirst.\n\nSource: https://...",
            "Wikipedia: Y\n\nSecond.\n\nSource: https://...",
        ]
        mock_client = mock_cls.return_value
        mock_client.messages.create.side_effect = [
            self._make_tool_response([self._make_tool_block("t1", "Topic X")]),
            self._make_tool_response([self._make_tool_block("t2", "Topic Y")]),
            self._make_end_response("Final answer."),
        ]
        result = agent.ask_agent("Multi-hop question.")
        self.assertEqual(result["num_searches"], 2)
        self.assertEqual(result["tools_called"], ["Topic X", "Topic Y"])

    @patch("agent.search_wikipedia")
    @patch("agent.anthropic.Anthropic")
    def test_no_tool_use_blocks_breaks_loop(self, mock_cls, mock_search):
        # stop_reason=tool_use but no tool_use blocks — must not loop forever
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Unexpected."
        spurious = MagicMock()
        spurious.stop_reason = "tool_use"
        spurious.content = [text_block]
        mock_client = mock_cls.return_value
        mock_client.messages.create.return_value = spurious
        result = agent.ask_agent("Edge case.")
        self.assertIsInstance(result, dict)
        self.assertEqual(mock_client.messages.create.call_count, 1)


# ---------------------------------------------------------------------------
# wikipedia.py — module structure
# ---------------------------------------------------------------------------

class TestWikipediaModule(unittest.TestCase):
    def test_cache_is_dict(self):
        self.assertIsInstance(wikipedia._cache, dict)

    def test_search_wikipedia_is_callable(self):
        self.assertTrue(callable(wikipedia.search_wikipedia))

    def test_search_wikipedia_has_query_parameter(self):
        sig = inspect.signature(wikipedia.search_wikipedia)
        self.assertIn("query", sig.parameters)

    def test_search_wikipedia_query_annotated_as_str(self):
        sig = inspect.signature(wikipedia.search_wikipedia)
        self.assertIs(sig.parameters["query"].annotation, str)


# ---------------------------------------------------------------------------
# eval.py — module structure
# ---------------------------------------------------------------------------

class TestEvalModule(unittest.TestCase):
    def test_test_cases_is_list(self):
        self.assertIsInstance(eval_module.TEST_CASES, list)

    def test_test_cases_starts_empty(self):
        self.assertEqual(len(eval_module.TEST_CASES), 0)

    def test_score_answer_correctness_callable(self):
        self.assertTrue(callable(eval_module.score_answer_correctness))

    def test_score_hallucination_callable(self):
        self.assertTrue(callable(eval_module.score_hallucination))

    def test_score_grounding_callable(self):
        self.assertTrue(callable(eval_module.score_grounding))

    def test_score_refusal_callable(self):
        self.assertTrue(callable(eval_module.score_refusal))

    def test_run_eval_callable(self):
        self.assertTrue(callable(eval_module.run_eval))

    def test_score_answer_correctness_signature(self):
        sig = inspect.signature(eval_module.score_answer_correctness)
        for param in ("question", "answer", "expected"):
            self.assertIn(param, sig.parameters)

    def test_score_hallucination_signature(self):
        sig = inspect.signature(eval_module.score_hallucination)
        for param in ("answer", "extracts"):
            self.assertIn(param, sig.parameters)

    def test_score_grounding_signature(self):
        sig = inspect.signature(eval_module.score_grounding)
        for param in ("answer", "extracts", "mode"):
            self.assertIn(param, sig.parameters)

    def test_run_eval_signature(self):
        sig = inspect.signature(eval_module.run_eval)
        self.assertIn("test_cases", sig.parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
