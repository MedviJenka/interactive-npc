from pathlib import Path
from crewai import Memory
from dataclasses import dataclass
from npc.factory import AgentFactory, NpcConfig

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


@dataclass
class Villagers:

    factory: AgentFactory

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
            knowledge_files=[KNOWLEDGE_DIR / "backstory.md"],
        )

        memories = self.factory.memory.recall(query,limit=5, scope="/conversation/marcus_holt")
        agent = self.factory.create_agent(config)
        prompt = f"Relevant memories: {memories} Player:{query}"
        response = agent.kickoff(prompt)

        self.factory.memory.remember(
            content=f"Player: {query} Marcus Holt: {response.raw}",
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
    villagers = Villagers(factory=factory)
    print(villagers.marcus_holt("where did you lose your finger?"))
