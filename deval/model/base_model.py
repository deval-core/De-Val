from abc import ABC, abstractmethod


class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def pull_model(self):
        ...

    @staticmethod
    @abstractmethod
    def submit_model(self):
        ...

    @staticmethod
    @abstractmethod
    def run(self):
        ... 