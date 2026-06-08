from settings import AgentInfra
from crewai import Agent, Crew, Task
from crewai import Memory
from crewai.project import CrewBase, agent, crew, task, after_kickoff, before_kickoff
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


knowledge_source = TextFileKnowledgeSource(file_paths=["backstory.txt"])

memory = Memory(recency_weight=0.4, semantic_weight=0.4, importance_weight=0.2, recency_half_life_days=14)


@CrewBase
class Npc(AgentInfra):

    @agent
    def agent(self) -> Agent:
        return Agent(config=self.agents_config['agent'], verbose=True, llm=self.llm, knowledge_sources=[knowledge_source], max_iter=1)

    @task
    def task(self) -> Task:
        return Task(config=self.tasks_config['task'])

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=False, memory=memory, stream=True)

    @before_kickoff
    def recall_context(self, inputs: dict) -> dict:
        query = inputs.get('query', '')
        recalled = memory.recall(query, limit=5, scope='/conversation')
        if recalled:
            past = '\n'.join(f"- {r.record.content}" for r in recalled)
        else:
            past = 'No prior conversations.'
        inputs['past_context'] = past
        memory.remember(content=f"Player said: {query}", scope='/conversation')
        return inputs

    @after_kickoff
    def remember(self, output):
        memory.remember(content=f"Marcus replied: {output.raw}", scope='/conversation')
        return output


if __name__ == '__main__':
    npc = Npc()
    response = npc.crew().kickoff({'query': 'what was my name?'})
    for each in response:
        print(each.content, end='')
