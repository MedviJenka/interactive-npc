from abc import ABC, abstractmethod
from dataclasses import dataclass
from crewai import Task, Agent
from crewai.project import CrewBase


@dataclass
class NpcFactory(ABC):

    name: str

    @abstractmethod
    def agent(self) -> Agent: ...

    @abstractmethod
    def task(self) -> Task: ...

    @abstractmethod
    def crew(self) -> CrewBase: ...
