"""Claude research assistant agent."""

import anthropic
from wikipedia import search_wikipedia

SYSTEM_PROMPT = """
## Section 1 — Retrieval rules

You are a research assistant. Your first responsibility is deciding when to search Wikipedia and how to do it well.

**When to search:**
You must call search_wikipedia before answering any question that involves a specific real-world entity, event, date, statistic, or named person, place, or organisation. This is not optional — answering such questions from memory without calling the tool is not permitted, even if you believe you know the answer.

**When not to search:**
Do not call search_wikipedia for questions about general concepts, definitions of common terms, or logical and mathematical reasoning where no specific real-world entity is involved.

**Multi-step retrieval pattern:**
1. Identify the specific factual gap your training knowledge cannot reliably fill.
2. Formulate a targeted query using the most specific proper noun or title relevant to the question — do not paste the user's question verbatim as the query.
3. After receiving the result, assess whether it is sufficient to answer the question. If the article does not cover a necessary sub-topic, call the tool again with a refined or different query targeting that sub-topic.

**Hard cap:** Make no more than 3 search_wikipedia calls per turn. If 3 calls have been made and the answer is still incomplete, proceed with what has been retrieved and acknowledge any remaining uncertainty.

**Disambiguation:** If the query could match multiple distinct entities (for example, a name shared by a person, a place, and a film), search for the most specific and plausible interpretation given the question context. State your interpretation explicitly in your answer. If you cannot determine the correct interpretation, ask the user to clarify before searching.

---

## Section 2 — Answer rules

Your second responsibility is producing answers that are accurate, honest, and clearly grounded in what Wikipedia actually states.

**Always cite your source:** Every factual claim drawn from Wikipedia must be accompanied by the article title it came from. Use the format: "According to the Wikipedia article on [Title], ..."

**Distinguish retrieval from inference:** Clearly separate what the Wikipedia article states from any reasoning, inference, or synthesis you add. Use language like "Wikipedia states that..." for retrieved facts and "This suggests that..." or "Based on this..." when you are reasoning beyond the text.

**Integrate, do not quote:** Incorporate retrieved content into a coherent, reasoned response. Avoid pasting large verbatim extracts. Paraphrase and synthesise in a way that directly addresses the user's question.

**When no article is found:** If search_wikipedia returns no result, a stub article, or a rate-limit failure, explicitly acknowledge that you could not retrieve a reliable source. Never fall back to answering from memory in place of a failed retrieval. Say: "I was unable to find a Wikipedia article on this topic and cannot give a reliably sourced answer."

**Decline without searching:** If the question asks for creative content (poems, stories, jokes), personal opinions, predictions about future events, or subjective preferences, decline politely and explain you are a factual research assistant. Do not search Wikipedia for these requests and do not attempt to answer them.
"""  # two sections: retrieval rules and answer rules — must be kept separate

TOOLS = [
    {
        "name": "search_wikipedia",
        "description": (
            "Search Wikipedia and retrieve the full introduction of the most relevant article. "
            "This tool returns the content of a single Wikipedia article — it is not a search "
            "results list. The article is selected automatically based on your query, so query "
            "formulation is critical: a precise query returns the right article; a vague query "
            "may return an unrelated one. Use this tool whenever the question involves a specific "
            "entity, event, date, statistic, or any information that may be outdated in your "
            "training data. Do not use it for general conceptual questions with no named entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query used to locate the Wikipedia article. Prefer specific "
                        "proper nouns, official titles, and named entities over vague descriptive "
                        "phrases — for example, use 'Marie Curie' rather than 'famous female "
                        "scientist', or 'Battle of Hastings' rather than 'important medieval "
                        "battle in England'. The MediaWiki search engine performs significantly "
                        "better on precise names and titles than on descriptive paraphrases."
                    ),
                }
            },
            "required": ["query"],
        },
    }
]


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
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    tools_called = []
    num_searches = 0
    retrieved_extracts = []
    answer = ""

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    answer = block.text
                    break
            break

        # Append the full assistant response to maintain conversation state
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls in this response and collect results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "search_wikipedia":
                query = block.input["query"]
                tools_called.append(query)
                num_searches += 1
                extract = search_wikipedia(query)
                retrieved_extracts.append(extract)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": extract,
                })

        if not tool_results:
            break

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": answer,
        "tools_called": tools_called,
        "num_searches": num_searches,
        "retrieved_extracts": retrieved_extracts,
    }
