from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from agents.llm_config import get_llm_settings


@tool
def extract_task_from_input(input_text: str) -> str:
    """Extract the core task from user input."""
    return f"Extracted task from input: {input_text.strip()}"


def create_langchain_agent():
    s = get_llm_settings()
    llm = ChatOpenAI(model=s["model"], base_url=s["base_url"], api_key=s["api_key"], temperature=0)

    agent = create_agent(
        model=llm,
        tools=[extract_task_from_input],
        system_prompt=(
            "You extract the core task from user input. Call the extract_task_from_input tool, "
            "then reply with ONLY the core task as one short sentence, nothing else."
        ),
    )
    return agent
