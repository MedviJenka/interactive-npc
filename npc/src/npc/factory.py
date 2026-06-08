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


class AgentFactory:
    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    @cached_property
    def llm(self) -> LLM:
        return LLM(
            model=Config.OPENAI_MODEL,
            api_key=Config.OPENAI_API_KEY,
        )

    def create_agent(self, config: NpcConfig) -> Agent:
        return Agent(
            role=config.role,
            goal=config.goal,
            backstory=config.backstory,
            verbose=config.verbose,
            max_iter=config.max_iter,
            llm=self.llm,
            knowledge_sources=self._knowledge_sources(
                config.knowledge_files
            ),
        )

    @staticmethod
    def _knowledge_sources(
        files: list[str],
    ) -> list[TextFileKnowledgeSource]:
        return [
            TextFileKnowledgeSource(file_paths=[file])
            for file in files
        ]


class VillageNpcs:
    def __init__(self, factory: AgentFactory) -> None:
        self.factory = factory

    def marcus_holt(self, query: str) -> str:
        config = NpcConfig(
            name="Marcus Holt",
            role="Sad NPC",
            goal=(
                "Hold short conversations using sad, tired, "
                "emotionally closed sentences."
            ),
            backstory=(
                "Marcus Holt is a war veteran in his forties. "
                "He rarely smiles and keeps most thoughts to himself."
            ),
            knowledge_files=["backstory.txt"],
        )

        memories = self.factory.memory.recall(
            query,
            limit=5,
            scope="/conversation/marcus_holt",
        )

        agent = self.factory.create_agent(config)

        prompt = f"""
Relevant memories:
{memories}

Player:
{query}
"""

        response = agent.kickoff(prompt)

        self.factory.memory.remember(
            content=(
                f"Player: {query}\n"
                f"Marcus Holt: {response.raw}"
            ),
            scope="/conversation/marcus_holt",
        )

        return response.raw


def create_memory() -> Memory:
    return Memory(
        recency_weight=0.4,
        semantic_weight=0.4,
        importance_weight=0.2,
        recency_half_life_days=14,
    )


if __name__ == "__main__":
    memory = create_memory()
    factory = AgentFactory(memory=memory)
    villagers = VillageNpcs(factory=factory)
    print(villagers.marcus_holt("What animal do I like?"))
