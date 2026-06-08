from dataclasses import dataclass
from functools import cached_property
from crewai import Agent, LLM, Memory
from pydantic import BaseModel, Field
from settings import Config
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


class NpcConfig(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    knowledge_files: list[str] = Field(default_factory=list)
    max_iter: int = 1
    verbose: bool = Config.VERBOSE


@dataclass
class AgentFactory:

    memory: Memory

    @cached_property
    def llm(self) -> LLM:
        return LLM(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_API_KEY)

    def create_agent(self, config: NpcConfig) -> Agent:
        return Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            verbose=config.verbose,
            max_iter=config.max_iter,
            llm=self.llm,
            knowledge_sources=self._knowledge_sources(config.knowledge_files)
        )

    @staticmethod
    def _knowledge_sources(files: list[str]) -> list[TextFileKnowledgeSource]:
        return [
            TextFileKnowledgeSource(file_paths=[file])
            for file in files
        ]
