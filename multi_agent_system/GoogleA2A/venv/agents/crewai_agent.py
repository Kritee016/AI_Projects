import os

from crewai import Crew, Agent, Task, LLM
from crewai.tools import BaseTool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from agents.llm_config import get_llm_settings


class ToolInput(BaseModel):
    question: str = Field(description="search query")


class SearchTool(BaseTool):
    name: str = "Search tool"
    description: str = "use this tool to find current information required for query"
    args_schema: type = ToolInput

    def _run(self, question: str):
        """use this tool to find current information required for query"""
        web_search = TavilySearchResults(tavily_api_key=os.environ["TAVILY_API_KEY"])
        result = web_search.invoke({"query": question})
        return str(result)


def create_crewai_agent(task_description: str):
    s = get_llm_settings()
    llm = LLM(
        model=f"openai/{s['model']}",  # "openai/" = generic OpenAI-compatible endpoint
        api_key=s["api_key"],
        base_url=s["base_url"],
        max_tokens=3000,
    )

    researcher = Agent(
        role="Researcher",
        goal="Search the web and gather information",
        backstory="Expert in web research using advanced techniques",
        tools=[SearchTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    task = Task(
        description=f"Research this briefly and return 5 bullet points only: {task_description}",
        expected_output="5 short bullet points with the most relevant findings",
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], verbose=True)
