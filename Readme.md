# RAG using Data Science University files #
## Quick run ##
### Preparation ###
Download: https://ollama.com/download <br>
In a terminal, write `ollama pull gemma:2b`

### Start ###
In a terminal, write `ollama run gemma:2b` to run the LLM which will be accessed by the program.<br>
Run Chatbot.py <br>
Make sure it's set to `myLLM = OllamaLLMInterface()`. You can try out a different LLM here too. If you want to try an external LLM, like Azure OpenAI, you'll need to create a configSecret.py with the relevant keys.