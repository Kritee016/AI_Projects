"""Single place that decides which LLM provider every agent uses.

All providers below expose an OpenAI-compatible API, so no paid OpenAI/Azure
key is required. Pick one with LLM_PROVIDER in your .env file.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Loads GoogleA2A/venv/.env (next to main.py)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PROVIDERS = {
    # Free tier, key from https://console.groq.com/keys
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
    },
    # Free tier, key from https://aistudio.google.com/app/apikey
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
    },
    # Fully local and free, install from https://ollama.com then `ollama pull llama3.1`
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.1",
        "key_env": None,
    },
}


def get_llm_settings() -> dict:
    name = os.getenv("LLM_PROVIDER", "groq").lower()
    if name not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got '{name}'")
    cfg = PROVIDERS[name]
    api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
    if not api_key:
        raise RuntimeError(f"Set {cfg['key_env']} in .env for provider '{name}'")
    return {
        "base_url": cfg["base_url"],
        "api_key": api_key,
        "model": os.getenv("LLM_MODEL", cfg["model"]),
    }
