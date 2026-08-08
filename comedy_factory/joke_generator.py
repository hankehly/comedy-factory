"""Joke generator workflow (see README.md for the flowchart).

Step 1 — Scan news for topics: agent step that uses Gemini's native Google
Search grounding to find real, current stories and returns them as plain
factual one-sentence topics.
"""

import re

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

from comedy_factory.settings import settings

_scan_news_agent = Agent(settings.model, capabilities=[WebSearch()])


def scan_news(num_topics: int = 5) -> list[str]:
    """Return `num_topics` factual news topics suitable for joke writing."""
    prompt = (settings.prompts_dir / "scan-news.md").read_text()
    prompt = prompt.replace("{NUM_TOPICS}", str(num_topics))

    result = _scan_news_agent.run_sync(prompt)

    topics = []
    for line in result.output.splitlines():
        # The prompt forbids numbering/bullets, but strip them if they slip through.
        topic = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if topic:
            topics.append(topic)
    return topics


def main():
    """Execute the joke generator workflow."""
    topics = scan_news(num_topics=5)
    for topic in topics:
        print(topic)


if __name__ == "__main__":
    main()
