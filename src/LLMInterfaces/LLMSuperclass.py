from abc import ABC, abstractmethod

class LLMSuperclass(ABC):     #Inherit from Abstract Base Class
    def __init__(self):
        pass

    def getName(self):
        return "Unspecified LLM"

    def setPrompt(self, promptStr: str):
        self.promptStr = promptStr

    @staticmethod
    @abstractmethod
    def isAvailable() -> bool:
        pass

    @abstractmethod
    def respond(self, query: str, context_text: str) -> str:
        pass