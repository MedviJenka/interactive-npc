from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from crewai import Agent, LLM, Memory
from pydantic import BaseModel, Field
from settings import Config


class NpcConfig(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    knowledge_files: list[Path] = Field(default_factory=list)
    max_iter: int = 1
    verbose: bool = Config.VERBOSE


@dataclass
class AgentFactory:

    memory: Memory

    @cached_property
    def llm(self) -> LLM:
        return LLM(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_API_KEY)

    @staticmethod
    def _read_md_file(file: Path) -> str:
        with open(file, 'r', encoding='utf-8') as f:
            return f.read()

    def create_agent(self, config: NpcConfig) -> Agent:
        knowledge = "\n\n".join(AgentFactory._read_md_file(f) for f in config.knowledge_files)
        backstory = f"{config.backstory}\n\n{knowledge}"

        return Agent(
            role=config.role,
            goal=config.goal,
            backstory=backstory,
            verbose=config.verbose,
            max_iter=config.max_iter,
            llm=self.llm,
        )
