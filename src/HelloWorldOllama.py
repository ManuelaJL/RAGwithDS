# Need to do  ollama run tinyllama at minimum, see TinyOllamaChatbot.py

from langchain_community.llms import Ollama

llm = Ollama(model="tinyllama")
print(llm.invoke("What is 2 + 2?"))