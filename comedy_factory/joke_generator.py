"""Joke generator workflow (see README.md for the flowchart).

Step 1 — Scan news for topics: agent step that uses Gemini's native Google
Search grounding to find real, current stories and returns them as plain
factual one-sentence topics.

Step 2 — Generate subtext: LLM step that turns a topic into a subtext — the
writer's opinion, the idea the eventual joke communicates.
"""

import re

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

from comedy_factory.prompts import load_prompt
from comedy_factory.settings import settings

_scan_news_agent = Agent(settings.model, capabilities=[WebSearch()])
_generate_subtext_agent = Agent(settings.model)


def scan_news(num_topics: int = 1) -> list[str]:
    """Return `num_topics` factual news topics suitable for joke writing."""
    prompt = load_prompt("scan-news.md", NUM_TOPICS=num_topics)

    result = _scan_news_agent.run_sync(prompt)

    topics = []
    for line in result.output.splitlines():
        # The prompt forbids numbering/bullets, but strip them if they slip through.
        topic = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if topic:
            topics.append(topic)
    return topics


def generate_subtext(topic: str) -> str:
    """Return a subtext — the writer's opinion about a news topic."""
    prompt = load_prompt("generate-subtext.md", TOPIC=topic)

    result = _generate_subtext_agent.run_sync(prompt)

    # The prompt forbids surrounding quotes, but strip them if they slip through.
    return result.output.strip().strip('"“”')


def main():
    """Execute the joke generator workflow."""
    topic = scan_news(num_topics=1)[0]
    subtext = generate_subtext(topic)
    print(f"Topic: {topic}")
    print(f"Subtext: {subtext}")


if __name__ == "__main__":
    main()
