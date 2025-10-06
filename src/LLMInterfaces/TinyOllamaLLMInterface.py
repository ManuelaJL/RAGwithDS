from LLMInterfaces.LLMSuperclass import LLMSuperclass
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import requests

class TinyOllamaLLMInterface(LLMSuperclass):

    @staticmethod
    def isAvailable():
        host="http://localhost:11434"
        try:
            response = requests.get(host)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def getName(self):
        return "TinyOllama"
    def respond(self, query: str, context_text: str) -> str:
        prompt = PromptTemplate.from_template(self.promptStr +
                                              """
                                              
                                Question:
                                {question}
                                
                                Context:
                                {context}
                                              """
                                              )
        llm = Ollama(model="tinyllama")
        qa_chain_without_retriever = prompt | llm

        return qa_chain_without_retriever.invoke({
            "question": query,
            "context":  context_text
        })
