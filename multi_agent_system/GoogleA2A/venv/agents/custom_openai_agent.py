from openai import OpenAI

from agents.llm_config import get_llm_settings


def generate_response(prompt: str) -> str:
    """Chat-completion call through the OpenAI SDK against any OpenAI-compatible endpoint."""
    s = get_llm_settings()
    client = OpenAI(base_url=s["base_url"], api_key=s["api_key"])
    response = client.chat.completions.create(
        model=s["model"],
        messages=[
            {"role": "system", "content": "You write clear, human-like answers using the provided research."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content
