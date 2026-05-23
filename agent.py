"""Claude research assistant agent."""

SYSTEM_PROMPT = ""  # two sections: retrieval rules and answer rules — must be kept separate

TOOLS = []


def ask_agent(question: str) -> dict:
    """Send a question to the Claude agent and return a structured response.

    The agent has access to the search_wikipedia tool and will call it as
    needed to ground its answer before responding.

    Args:
        question: The user's question in natural language.

    Returns:
        A dict with the following keys:

        {
            "answer": str,               # the agent's final response text
            "tools_called": list[str],   # names of every tool called, in order
            "num_searches": int,         # number of search_wikipedia calls made
            "retrieved_extracts": list[str]  # the raw extract text returned by
                                             # each search_wikipedia call, in order
        }

        Note: retrieved_extracts contains the extract from every Wikipedia
        lookup made during the turn, not just the first. The eval scorer
        requires all extracts to check factual grounding across every article
        retrieved, not only the article that contributed to the final answer.
    """
    pass
