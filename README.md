# Claude + Wikipedia Q&A System

A research assistant that answers questions by retrieving Wikipedia articles via the MediaWiki REST API and grounding responses using Claude (`claude-sonnet-4-6`). Built with the Anthropic Python SDK and a custom `search_wikipedia` tool — no hosted RAG layer or third-party retrieval service.

---

## Features

- Grounded answers: Claude must call `search_wikipedia` before answering any factual question
- Multi-step retrieval: up to 3 Wikipedia lookups per question, with a mandatory fallback if none were made
- Graceful refusals: out-of-scope questions (poems, opinions, predictions) are declined without searching
- LLM-as-judge eval suite: 35 hand-authored test cases across 6 categories, scored on 5 metrics

---

## Project Structure

```
.
├── main.py               # CLI entrypoint (normal / demo / eval modes)
├── agent.py              # Claude agent with agentic tool-use loop
├── wikipedia.py          # MediaWiki REST API integration
├── eval.py               # Test cases and LLM-as-judge scoring functions
├── agent_mock_test.py    # Mock tests for agent.py
├── wikipedia_mock_test.py# Mock tests for wikipedia.py
├── eval_mock_test.py     # Mock tests for eval.py
└── CLAUDE.md             # Project plan and implementation learnings
```

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:prai-cmd/wiki_claude.git
cd wiki_claude
```

**2. Install dependencies**
```bash
pip install anthropic requests
```

**3. Set your Anthropic API key**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

No API key is required for Wikipedia (MediaWiki REST API is public).

---

## Usage

### Normal mode — ask a single question
```bash
python main.py "Who was the first human to walk on the Moon?"
```

Example output:
```
============================================================
Question: Who was the first human to walk on the Moon?
------------------------------------------------------------
According to the Wikipedia article on Neil Armstrong, ...
------------------------------------------------------------
Wikipedia searches made: 1
Articles retrieved:
  - Neil Armstrong
============================================================
```

### Demo mode — run 3 sample questions
```bash
python main.py
```

Cycles through a factual, an ambiguous, and a multi-hop question with a 1-second pause between each.

### Eval mode — run the full evaluation suite
```bash
python main.py --eval
```

Runs all 35 test cases and prints a 5-metric summary table.

---

## Evaluation Metrics

| Metric | Target | Last run |
|---|---|---|
| Answer correctness (1–5) | ≥ 4.0 | 4.7 ✅ |
| Hallucination rate | < 0.05 | 0.0 ✅ |
| Factual grounding | > 0.90 | 0.73 ❌ |
| Refusal rate | > 0.90 | 1.0 ✅ |
| Memory answer rate | = 0.0 | 0.0 ✅ |

**Grounding** is the main open challenge: comparative and ambiguous questions require synthesis across concepts, and the scorer marks any claim absent from the retrieved extract as ungrounded — even when correct.

---

## Architecture Notes

- `tool_choice={"type": "any"}` is set on the first API call to force tool use and eliminate memory answers. Out-of-scope questions are detected via a keyword list and use `{"type": "auto"}` instead, allowing the model to decline gracefully.
- A mandatory fallback search runs after the agentic loop if no searches were made. A noun phrase is extracted from the question, Wikipedia is queried, and Claude is asked to revise its answer citing the new source.
- The system prompt is split into two sections — retrieval rules and answer rules — to make iteration on one independent of the other.
- The LLM-as-judge evaluator is a direct Claude API call with separate scoring prompts for correctness, hallucination, and grounding.

---

## Running Mock Tests

```bash
python agent_mock_test.py
python wikipedia_mock_test.py
python eval_mock_test.py
```
