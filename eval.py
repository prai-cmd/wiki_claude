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
TEST_CASES = [
    # ------------------------------------------------------------------
    # factual (10) — direct who/what/when/where questions
    # ------------------------------------------------------------------
    {
        "question": "When was the Eiffel Tower constructed?",
        "category": "factual",
        "expected_answer": (
            "The Eiffel Tower was constructed between 1887 and 1889 as the entrance arch "
            "for the 1889 World's Fair in Paris."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "Who wrote the novel Nineteen Eighty-Four?",
        "category": "factual",
        "expected_answer": (
            "Nineteen Eighty-Four was written by George Orwell and published in 1949."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "What is the capital city of Australia?",
        "category": "factual",
        "expected_answer": "The capital city of Australia is Canberra.",
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "How many bones are in the adult human body?",
        "category": "factual",
        "expected_answer": "The adult human body has 206 bones.",
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "In what year did the Berlin Wall fall?",
        "category": "factual",
        "expected_answer": (
            "The Berlin Wall fell in 1989, specifically on November 9, 1989."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "Who was the first human to walk on the Moon?",
        "category": "factual",
        "expected_answer": (
            "Neil Armstrong was the first human to walk on the Moon on July 20, 1969, "
            "during the Apollo 11 mission."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "What is the speed of light in a vacuum?",
        "category": "factual",
        "expected_answer": (
            "The speed of light in a vacuum is exactly 299,792,458 metres per second."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "In what year was the United Nations founded?",
        "category": "factual",
        "expected_answer": (
            "The United Nations was founded in 1945, with the UN Charter coming into "
            "force on October 24, 1945."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "What is the largest ocean on Earth?",
        "category": "factual",
        "expected_answer": (
            "The Pacific Ocean is the largest and deepest ocean on Earth."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "Who painted the Mona Lisa?",
        "category": "factual",
        "expected_answer": (
            "The Mona Lisa was painted by Italian Renaissance artist Leonardo da Vinci, "
            "most likely between 1503 and 1519."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    # ------------------------------------------------------------------
    # comparative (5) — difference/contrast between two entities
    # ------------------------------------------------------------------
    {
        "question": "What is the difference between DNA and RNA?",
        "category": "comparative",
        "expected_answer": (
            "DNA is double-stranded, contains deoxyribose sugar, uses thymine, and serves "
            "as the long-term store of genetic information. RNA is typically single-stranded, "
            "contains ribose sugar, uses uracil instead of thymine, and plays roles in "
            "protein synthesis and gene regulation."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "How do black holes and neutron stars differ?",
        "category": "comparative",
        "expected_answer": (
            "Both are remnants of massive stars. Neutron stars are extremely dense objects "
            "supported by neutron degeneracy pressure with a radius of roughly 10 km and a "
            "definite surface. Black holes form when mass exceeds the Tolman-Oppenheimer-Volkoff "
            "limit; they have no surface, are defined by an event horizon, and their gravity "
            "prevents even light from escaping."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "What distinguishes the Roman Republic from the Roman Empire?",
        "category": "comparative",
        "expected_answer": (
            "The Roman Republic (509–27 BC) was governed by elected magistrates and a Senate "
            "with power distributed among citizens. The Roman Empire (27 BC onwards) was ruled "
            "by emperors with concentrated autocratic authority, beginning with Augustus Caesar."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "How does photosynthesis differ from cellular respiration?",
        "category": "comparative",
        "expected_answer": (
            "Photosynthesis converts light energy, water, and carbon dioxide into glucose and "
            "oxygen in plants and algae. Cellular respiration converts glucose and oxygen into "
            "carbon dioxide, water, and ATP energy in most living organisms. They are broadly "
            "reverse processes in terms of reactants and products."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "What are the key differences between viruses and bacteria?",
        "category": "comparative",
        "expected_answer": (
            "Bacteria are single-celled living organisms with their own metabolism and can "
            "reproduce independently. Viruses are not considered fully alive; they lack cellular "
            "structure, have no independent metabolism, and can only replicate inside host cells. "
            "Bacteria can be treated with antibiotics; viruses cannot."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    # ------------------------------------------------------------------
    # multi_hop (5) — answer requires synthesising two distinct articles
    # ------------------------------------------------------------------
    {
        "question": (
            "What country was Marie Curie born in, and what element did she "
            "co-discover with her husband Pierre?"
        ),
        "category": "multi_hop",
        "expected_answer": (
            "Marie Curie was born in Warsaw, in what is now Poland. She and Pierre Curie "
            "co-discovered the elements polonium and radium."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": (
            "Which Shakespeare play features the character Ophelia, "
            "and what genre is that play?"
        ),
        "category": "multi_hop",
        "expected_answer": (
            "Ophelia appears in Hamlet. Hamlet is a tragedy written by William Shakespeare."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "What is the capital of the country that won the 2018 FIFA World Cup?",
        "category": "multi_hop",
        "expected_answer": (
            "France won the 2018 FIFA World Cup. The capital of France is Paris."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "Who invented the telephone and what country was he born in?",
        "category": "multi_hop",
        "expected_answer": (
            "Alexander Graham Bell is credited with inventing the telephone. "
            "He was born in Edinburgh, Scotland."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": (
            "In what city is the headquarters of the organisation "
            "that awards the Nobel Peace Prize?"
        ),
        "category": "multi_hop",
        "expected_answer": (
            "The Nobel Peace Prize is awarded by the Norwegian Nobel Committee, "
            "which is based in Oslo, Norway."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    # ------------------------------------------------------------------
    # ambiguous (5) — query matches multiple distinct Wikipedia entities
    # ------------------------------------------------------------------
    {
        "question": "Tell me about Mercury.",
        "category": "ambiguous",
        "expected_answer": (
            "Mercury may refer to the planet Mercury (the smallest planet and closest to "
            "the Sun), the chemical element mercury (atomic number 80, a liquid metal at "
            "room temperature), or the Roman god Mercury. The agent should state which "
            "interpretation it used and why."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "What is Python?",
        "category": "ambiguous",
        "expected_answer": (
            "Python most commonly refers to the high-level, general-purpose programming "
            "language created by Guido van Rossum. It may also refer to pythons, the family "
            "of non-venomous constrictor snakes. The agent should declare which interpretation "
            "it searched for."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "Tell me about Apollo.",
        "category": "ambiguous",
        "expected_answer": (
            "Apollo may refer to the Apollo space program (NASA's lunar program that landed "
            "humans on the Moon), the Greek and Roman deity Apollo, or several other uses "
            "including the Apollo asteroid group. The agent should state its interpretation."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "Tell me about Genesis.",
        "category": "ambiguous",
        "expected_answer": (
            "Genesis may refer to the Book of Genesis (the first book of the Hebrew Bible "
            "and Christian Old Testament) or the English rock band Genesis formed in 1967. "
            "The agent should clarify which interpretation it used."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "Who or what is Victoria?",
        "category": "ambiguous",
        "expected_answer": (
            "Victoria may refer to Queen Victoria (Queen of the United Kingdom 1837–1901), "
            "the state of Victoria in Australia, Lake Victoria in Africa, or other uses. "
            "The agent should specify which interpretation it pursued."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    # ------------------------------------------------------------------
    # out_of_scope (5) — Wikipedia cannot answer; agent must decline
    # ------------------------------------------------------------------
    {
        "question": "What will the stock price of Apple Inc. be tomorrow?",
        "category": "out_of_scope",
        "expected_answer": (
            "This question asks for a future stock price prediction, which is out of scope "
            "for a Wikipedia-based research assistant. The agent should decline without "
            "fabricating a figure."
        ),
        "grounding_mode": "strict",
        "requires_tool": False,
    },
    {
        "question": "Write me a short poem about the ocean.",
        "category": "out_of_scope",
        "expected_answer": (
            "This is a creative writing request and is out of scope for a Wikipedia-based "
            "research assistant. The agent should decline."
        ),
        "grounding_mode": "strict",
        "requires_tool": False,
    },
    {
        "question": "What is the best pizza topping?",
        "category": "out_of_scope",
        "expected_answer": (
            "This is a matter of personal opinion with no factual Wikipedia answer. "
            "The agent should decline rather than state a preference."
        ),
        "grounding_mode": "strict",
        "requires_tool": False,
    },
    {
        "question": "What is my favourite colour?",
        "category": "out_of_scope",
        "expected_answer": (
            "This question refers to personal information about the user that the agent "
            "has no access to. The agent should decline."
        ),
        "grounding_mode": "strict",
        "requires_tool": False,
    },
    {
        "question": "Can you predict who will win the next Olympic 100m final?",
        "category": "out_of_scope",
        "expected_answer": (
            "Predicting future sporting outcomes is out of scope for a Wikipedia-based "
            "research assistant. The agent should decline without speculating."
        ),
        "grounding_mode": "strict",
        "requires_tool": False,
    },
    # ------------------------------------------------------------------
    # coverage_edge (5) — information exists on Wikipedia but may be
    # stale, rapidly changing, or at the boundary of article coverage
    # ------------------------------------------------------------------
    {
        "question": "What is the current population of the Tokyo metropolitan area?",
        "category": "coverage_edge",
        "expected_answer": (
            "The Greater Tokyo Area is the most populous metropolitan area in the world "
            "with a population of approximately 37–38 million as of recent census figures. "
            "The exact current figure may differ from Wikipedia's most recent data."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "Who is the current Secretary-General of the United Nations?",
        "category": "coverage_edge",
        "expected_answer": (
            "As of the time of writing, António Guterres is the Secretary-General of the "
            "United Nations, having taken office on January 1, 2017. This may be outdated "
            "if a successor has been appointed."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "How many confirmed exoplanets have been discovered?",
        "category": "coverage_edge",
        "expected_answer": (
            "As of recent counts, over 5,500 exoplanets have been confirmed. The number "
            "increases frequently as new discoveries are announced, so Wikipedia's figure "
            "may not reflect the most current total."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
    {
        "question": "What is the world record for the men's 100 metres sprint?",
        "category": "coverage_edge",
        "expected_answer": (
            "The men's 100 metres world record is 9.58 seconds, set by Usain Bolt of "
            "Jamaica at the 2009 World Athletics Championships in Berlin."
        ),
        "grounding_mode": "strict",
        "requires_tool": True,
    },
    {
        "question": "What is the tallest building in the world?",
        "category": "coverage_edge",
        "expected_answer": (
            "As of the most recent data, the Burj Khalifa in Dubai, United Arab Emirates, "
            "is the tallest building in the world, standing at 828 metres (2,717 feet). "
            "This ranking could change if a taller structure is completed."
        ),
        "grounding_mode": "lenient",
        "requires_tool": True,
    },
]


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
