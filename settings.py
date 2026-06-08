from crewai import LLM, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from pydantic import Field
from functools import cached_property
from pydantic_settings import BaseSettings
from dataclasses import dataclass


class Settings(BaseSettings):
    API_VERSION:           str = Field(default='v1')
    ENV:                   str = Field(default='dev')
    LOG_DIR:               str = Field(default='/logs')
    LOG_LEVEL:             str = Field(default='INFO')
    VERBOSE:               str = Field(default=True)
    ANTHROPIC_API_KEY:     str = Field(...)
    LOGFIRE_TOKEN:         str = Field(...)
    MCP_PROXY_AUTH_TOKEN:  str = Field(...)
    CLAUDE_MODEL:          str = Field(...)
    DANGEROUSLY_OMIT_AUTH: str = Field(...)
    GHCR_TOKEN:            str = Field(...)
    OPENAI_API_KEY:        str = Field(...)


Config = Settings()


@dataclass
class AgentInfra:

    agents_config: dict = 'config/agents.yaml'
    tasks_config: dict = 'config/tasks.yaml'
    agents: list[BaseAgent] = None
    tasks: list[Task] = None

    @cached_property
    def llm(self) -> LLM:
        return LLM(model='gpt-5.2', api_key=Config.OPENAI_API_KEY)
