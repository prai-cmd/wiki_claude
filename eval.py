"""Test cases and scoring functions."""

# Expected shape of each entry in TEST_CASES:
# {
#     "question": str,
#     "category": str,  # factual | comparative | multi_hop | ambiguous |
#                       # out_of_scope | coverage_edge
#     "expected_answer": str,
#     "grounding_mode": str,  # strict | lenient
#     "requires_tool": bool
# }
TEST_CASES = []


def score_answer_correctness(question: str, answer: str, expected: str) -> float:
    """Score the factual correctness of an answer via a direct Claude API call.

    Sends question, answer, and expected_answer to Claude with a scoring
    rubric and returns a float on the 1.0–5.0 scale, where 1.0 is fully
    incorrect and 5.0 is fully correct and complete.

    The Claude call is made directly via the Anthropic SDK — no hosted
    eval framework or third-party eval service is used.

    Args:
        question: The original question posed to the agent.
        answer: The agent's response to be scored.
        expected: The reference answer to score against.

    Returns:
        A float in the range [1.0, 5.0].
    """
    pass


def score_hallucination(answer: str, extracts: list) -> float:
    """Score the hallucination rate of an answer against retrieved extracts.

    Computes the fraction of claims in the answer that actively contradict
    any of the retrieved Wikipedia extracts — i.e., the extract asserts X
    and the answer asserts not-X.

    This metric is independent from factual grounding. A claim can be
    ungrounded (absent from all extracts) without being a hallucination.
    Silence in the extract is not contradiction. The scorer must apply
    distinct criteria from score_grounding and never conflate absence
    with contradiction.

    Args:
        answer: The agent's response to be scored.
        extracts: List of raw extract strings returned by each
            search_wikipedia call during the turn.

    Returns:
        A float in [0.0, 1.0]: fraction of claims that contradict an extract.
        Target: below 0.05.
    """
    pass


def score_grounding(answer: str, extracts: list, mode: str) -> float:
    """Score the factual grounding of an answer against retrieved extracts.

    Computes the fraction of claims in the answer that are positively
    traceable to the retrieved Wikipedia extracts.

    Scoring modes:
        strict  — every claim must be directly traceable to the extract
                  text. A claim the extract does not mention fails grounding
                  even if it is independently true.
        lenient — claims traceable to any of the retrieved extracts pass.
                  Reasonable inference from extract content is acceptable,
                  but fabricated or training-knowledge-only claims still fail.

    This metric is independent from hallucination scoring. Ungrounded claims
    are not the same as contradictory claims and must be scored separately.

    Args:
        answer: The agent's response to be scored.
        extracts: List of raw extract strings returned by each
            search_wikipedia call during the turn.
        mode: "strict" or "lenient".

    Returns:
        A float in [0.0, 1.0]: fraction of claims that pass grounding.
        Target: above 0.90.
    """
    pass


def score_refusal(answer: str) -> float:
    """Score whether the agent correctly declined an out-of-scope question.

    Args:
        answer: The agent's response to be scored.

    Returns:
        1.0 if the answer declines to answer without hallucinating content,
        0.0 otherwise.
    """
    pass


def run_eval(test_cases: list) -> dict:
    """Run the full evaluation suite over a list of test cases.

    Routes each test case to the appropriate scorer(s) based on its
    category field:
        - factual, comparative, multi_hop, ambiguous: scored on
          answer_correctness, hallucination, and grounding.
        - out_of_scope: scored on refusal rate only.
        - coverage_edge: scored on answer_correctness and grounding only.

    Rate-limit exclusions:
        Any case where the agent's answer contains the string
        "RATE_LIMIT_FAILURE" is excluded from all metric denominators.
        Excluded cases are logged separately with their count so the
        exclusion is visible in results and not silently absorbed.

    Args:
        test_cases: List of test case dicts conforming to the TEST_CASES
            shape (question, category, expected_answer, grounding_mode,
            requires_tool).

    Returns:
        A dict containing aggregated metric scores, per-case breakdowns,
        and the count of cases excluded due to RATE_LIMIT_FAILURE.
    """
    pass
