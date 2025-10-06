from LLMInterfaces.LLMSuperclass import LLMSuperclass
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import requests

class OllamaLLMInterface(LLMSuperclass):
    model = "gemma:2b"

    @staticmethod
    def isAvailable():
        host="http://localhost:11434"
        try:
            response = requests.get(host)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def getName(self):
        return f"Ollama with {self.model}"

    def respond(self, query: str, context_text: str) -> str:
        prompt = PromptTemplate.from_template(self.promptStr +
                                              """
                                              
                                Question:
                                {question}
                                
                                Context:
                                {context}
                                              """
                                              )
        llm = Ollama(model=self.model)
        qa_chain_without_retriever = prompt | llm

        return qa_chain_without_retriever.invoke({
            "question": query,
            "context":  context_text
        })



# https://ollama.com/download
# Installation automatically extracted files to: C:\Users\manue\AppData\Local\Programs\Ollama\
# To download the model: ollama pull gemma:latest
# Run it on GPU: ollama run gemma
# To make it run on cpu instead: (and use powershell as administrator) type into terminal:
#   set OLLAMA_NUM_GPU_LAYERS=0
#   set CUDA_VISIBLE_DEVICES=
#   ollama run gemma
# I got: Error: llama runner process has terminated: cudaMalloc failed: out of memory
# So, created C:\Users\manue\.ollama\config to force it to use cpu. But still same error.
# Tried instead: Ollama pull gemma:2b, then ollama run gemma:2b and that worked
# ollama run gemma:2b --cpu-only