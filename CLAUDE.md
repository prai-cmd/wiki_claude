# Plan: Claude + Wikipedia Q&A System

---

## 1. System Prompt and Tool Definition

The system prompt should establish two things clearly: the model's role and the expected reasoning pattern for using the tool. To be explicit: `search_wikipedia` is a custom-built tool we define and pass to Claude via the tools API — it is not a built-in or hosted search capability like Anthropic's `web_search` tool type, OpenAI browsing, or Perplexity.

The role framing should position the model as a research assistant that prioritizes accuracy over speed. It should be instructed to never answer factual questions from its training knowledge alone when a Wikipedia lookup could verify or expand on that answer — this reduces hallucination on specific facts (dates, names, statistics, definitions) while keeping the model's reasoning capabilities intact for synthesis and explanation.

The tool use guidance in the system prompt should enforce a deliberate, multi-step retrieval pattern. Before calling the tool, the model should identify the specific factual gap in its current knowledge. It should formulate a targeted search query — not a verbatim copy of the user's question, but a refined query that isolates the entity or concept most likely to appear as a Wikipedia article title or section heading. After receiving results, the model should assess whether the content is sufficient or whether a follow-up search on a related concept is needed. This teaches the model to treat Wikipedia as a corpus to navigate, not just a single-shot oracle.

The prompt should also set answer quality expectations: responses should cite which Wikipedia article(s) informed the answer, integrate retrieved content with reasoning rather than just quoting it, and clearly distinguish between what the Wikipedia source states and any inferences or summaries the model adds. It should handle ambiguity gracefully — if a query could refer to multiple entities (e.g., "Mercury" the planet vs. the element vs. the band), the model should either ask for clarification or search for the most plausible interpretation and state its assumption.

The tool definition itself should describe `search_wikipedia` as accepting a natural-language or keyword query string and returning a summary of the most relevant Wikipedia article along with its title. The description should make clear that the tool returns article-level content, not search result lists, so the model understands it needs good query formulation upfront. The parameter description should coach the model to prefer specific proper nouns and titles over vague descriptive phrases, since the MediaWiki search engine performs better on those.

---

## 2. Wikipedia Retrieval Integration

The integration has two layers: search (finding the right article) and fetch (getting its content). This integration calls the MediaWiki REST API directly from our own code — there is no middleware, hosted RAG layer, or managed retrieval service sitting between our application and Wikipedia.

For search, the MediaWiki REST API exposes a search endpoint that accepts a query string and returns a ranked list of article titles with short extracts. The integration should call this endpoint first, take the top result's title, and proceed to fetch that article. It's worth implementing a lightweight fallback: if the top result's title looks like a disambiguation page (MediaWiki often includes "(disambiguation)" in the title or returns a page whose first paragraph lists multiple meanings), the integration should either return all candidate titles to the model so it can pick, or try the second-ranked result.

For fetch, once a title is confirmed, the integration should call the REST API's page summary endpoint, which returns a structured JSON object including the article's title, a plain-text extract of the introduction, and a URL. The extract length from this endpoint is typically one to several paragraphs, which is the right scope for the model — long enough to answer most factual questions, short enough not to flood the context window. For questions that require deeper article sections (e.g., a specific historical event within a long biography), a more advanced integration could fetch the full article and extract the most relevant section by matching section headings to the query, but the summary endpoint is a good starting point.

Error handling should cover: articles not found (return a clear "no article found" message so the model can decide to rephrase or acknowledge uncertainty), rate limiting (MediaWiki is generous but a small retry with backoff handles transient failures), and network timeouts. The tool should return a normalized response structure that always includes at minimum the article title, the extract text, and the source URL, even in partial-success cases.

No API key is required for read-only MediaWiki REST API access, which simplifies deployment. Requests should include a descriptive `User-Agent` header identifying the application, as MediaWiki's API policy requires this.

---

## 3. Eval Suite

The evaluation suite should cover three dimensions: retrieval quality, answer quality, and tool use behavior.

The suite tracks exactly five metrics, each with a hard target:

- **Answer correctness:** whether the final answer is factually correct, scored by an LLM judge on a 1–5 scale. Target: average of 4.0 or above.

- **Hallucination rate:** fraction of answers containing claims that actively contradict the retrieved Wikipedia content — the extract says X, the answer says not-X. Target: below 5%.

- **Factual grounding:** fraction of claims in the answer that are positively traceable to the retrieved extract. A claim the extract does not mention fails grounding even if it is true. Target: above 90%.

- **Memory answer rate:** how often the model answers without making any tool call. Target: 0%.

- **Graceful refusal rate:** how often the model correctly declines out-of-scope questions without hallucinating an answer. Target: above 90%.

Hallucination rate and factual grounding are independent metrics. A claim can be ungrounded without contradicting the extract — silence in the extract is not contradiction. The judge must score them separately using distinct criteria.

The LLM-as-judge evaluator is a direct Claude API call we write ourselves — not a hosted eval framework or third-party eval service like RAGAS, LangSmith, or similar.

The test case set should be stratified across question types: factual lookup (who, what, when, where), explanatory (how does X work), comparative (what is the difference between X and Y), and current-events-adjacent (recent enough that training data may be stale). These test cases are hand-authored or generated directly by us — they are not sourced from a pre-existing benchmark dataset or third-party test suite. Around 50–100 test cases across these categories gives enough signal for iteration without being prohibitively expensive to run. All test cases should have a human-written reference answer and a label indicating whether the answer is findable on Wikipedia, so the suite can separately score "findable" vs. "unfindable" cases — for unfindable questions, the expected behavior is an honest acknowledgment rather than a fabricated answer.

---

## 4. Implementation Learnings

### Agent Architecture

- `tool_choice={"type": "any"}` on the first API call is essential to eliminate memory answers (drove memory_answer_rate from 0.33 → 0.0). Without it, Claude frequently answers from training knowledge.
- Out-of-scope questions (poems, opinions, predictions) must be detected before applying `tool_choice={"type": "any"}` — forcing tool use on these prevents the model from declining gracefully and tanks refusal rate. Detection is via a keyword list (`poem`, `story`, `joke`, `opinion`, `predict`, `favourite`, `best`, `worst`, `will`, `should i`).
- A mandatory fallback search after the agentic loop handles cases where the model still skips tool use despite `tool_choice`. Extract a noun phrase from the question, call `search_wikipedia`, then ask Claude to revise its answer citing the new source.
- Noun-phrase extraction for fallback queries requires regex strip patterns (removing question preamble like "how many", "what is the") plus a stop-word filter. Verbatim question strings perform poorly as Wikipedia queries.

### System Prompt

- Split into two explicit sections: **Retrieval rules** (when/how to search) and **Answer rules** (how to cite and present). Keeping them separate makes it easier to iterate on one without regressing the other.
- Superlatives (largest, fastest, tallest, most) and comparative questions (X vs Y, difference between X and Y) must be explicitly called out as mandatory search triggers — the model will otherwise answer from memory on these.
- Disambiguation: instruct the model to never resolve an ambiguous query from memory. Always search first, then state the interpretation. Without this, ambiguous questions (Mercury, Apollo, Genesis) return ungrounded answers.
- Declining out-of-scope requests must be a separate explicit instruction. If the system prompt only says "when to search", the model may still attempt to answer creative or opinion questions.

### Eval Suite

- 35 test cases across 6 categories: `factual`, `comparative`, `multi_hop`, `ambiguous`, `out_of_scope`, `coverage_edge`.
- Best run results (as of last full eval):
  - Answer correctness: 4.7 ✅ (target ≥ 4.0)
  - Hallucination rate: 0.0 ✅ (target < 0.05)
  - Grounding: 0.73 ❌ (target > 0.90) — main remaining gap
  - Refusal rate: 1.0 ✅ (target > 0.90)
  - Memory answer rate: 0.0 ✅ (target = 0.0)
- Grounding is the hardest metric to improve: comparative and ambiguous questions require the model to synthesise across multiple concepts, and the grounding scorer marks any claim not present in the retrieved extract as ungrounded — even if the claim is factually correct and well-reasoned.
- `memory_answer_rate` must be tracked separately per `requires_tool` cases only — out-of-scope cases that correctly decline are not memory answers.

### CLI and Setup

- `main.py` provides three modes: normal (`python main.py "question"`), demo (no args, 3 hardcoded questions), eval (`--eval` flag).
- Check `ANTHROPIC_API_KEY` at startup before any imports of agent/eval — gives a clear actionable error rather than an SDK-level exception.
- No API key required for Wikipedia MediaWiki REST API access.
