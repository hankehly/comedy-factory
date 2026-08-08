"""Loading and templating of the prompt files in the prompts/ directory."""

from comedy_factory.settings import settings


def load_prompt(name: str, **placeholders: object) -> str:
    """Load a prompt file and substitute its `{PLACEHOLDER}` values."""
    prompt = (settings.prompts_dir / name).read_text()
    for key, value in placeholders.items():
        prompt = prompt.replace("{" + key + "}", str(value))
    return prompt
