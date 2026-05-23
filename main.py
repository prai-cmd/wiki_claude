#!/usr/bin/env python3
"""Runnable CLI for the Wikipedia Q&A system."""

import os
import sys
import time

SEP = "=" * 60
THIN = "-" * 60

DEMO_QUESTIONS = [
    ("Factual",   "Who was the first human to walk on the Moon?"),
    ("Ambiguous", "Tell me about Mercury."),
    ("Multi-hop", "What is the capital of the country that won the 2018 FIFA World Cup?"),
]


def parse_titles(extracts: list) -> list:
    titles = []
    for extract in extracts:
        first_line = extract.splitlines()[0] if extract else ""
        if first_line.startswith("Wikipedia: "):
            titles.append(first_line[len("Wikipedia: "):].strip())
    return titles


def print_result(question: str, response: dict) -> None:
    print(SEP)
    print(f"Question: {question}")
    print(THIN)
    print(response["answer"])
    print(THIN)
    print(f"Wikipedia searches made: {response['num_searches']}")
    titles = parse_titles(response["retrieved_extracts"])
    if titles:
        print("Articles retrieved:")
        for title in titles:
            print(f"  - {title}")
    else:
        print("Articles retrieved: none")
    print(SEP)


def run_normal(question: str) -> None:
    from agent import ask_agent
    response = ask_agent(question)
    print_result(question, response)


def run_demo() -> None:
    from agent import ask_agent
    print(SEP)
    print("DEMO MODE — running 3 sample questions")
    print(SEP)
    for i, (label, question) in enumerate(DEMO_QUESTIONS):
        print(f"\n[{label}]")
        try:
            response = ask_agent(question)
            print_result(question, response)
        except Exception as exc:
            print(f"Error on question '{question}': {exc}", file=sys.stderr)
        if i < len(DEMO_QUESTIONS) - 1:
            time.sleep(1)


def run_eval_mode() -> None:
    from eval import run_eval, TEST_CASES
    print(SEP)
    print("EVAL MODE — running full suite")
    print(f"Total test cases: {len(TEST_CASES)}")
    print(SEP)
    results = run_eval(TEST_CASES)
    print()
    print("METRIC SUMMARY")
    print(THIN)
    metrics = [
        ("Answer correctness", results["avg_answer_correctness"], ">=4.0",  lambda v: v is not None and v >= 4.0),
        ("Hallucination rate", results["avg_hallucination_rate"], "<0.05",  lambda v: v is not None and v < 0.05),
        ("Grounding",          results["avg_grounding"],          ">0.90",  lambda v: v is not None and v > 0.90),
        ("Refusal rate",       results["avg_refusal_rate"],       ">0.90",  lambda v: v is not None and v > 0.90),
        ("Memory answer rate", results["memory_answer_rate"],     "=0.0",   lambda v: v is not None and v == 0.0),
    ]
    for name, value, target, passes in metrics:
        status = "OK" if passes(value) else "FAIL"
        val_str = f"{value:.4f}" if value is not None else "N/A"
        print(f"  [{status}] {name:<22} {val_str:<10} [target {target}]")
    print(THIN)
    print(f"Total evaluated:       {results['total_evaluated']}")
    print(f"Excluded (rate limit): {results['excluded_rate_limit']}")
    print(SEP)


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        print("Set it with:", file=sys.stderr)
        print('  export ANTHROPIC_API_KEY="your-api-key-here"', file=sys.stderr)
        sys.exit(1)


def main() -> None:
    _check_api_key()
    args = sys.argv[1:]

    try:
        if "--eval" in args:
            run_eval_mode()
        elif len(args) == 0:
            run_demo()
        elif len(args) == 1:
            run_normal(args[0])
        else:
            print("Usage:", file=sys.stderr)
            print('  python main.py                  # demo mode', file=sys.stderr)
            print('  python main.py "question"       # normal mode', file=sys.stderr)
            print('  python main.py --eval           # eval mode', file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
