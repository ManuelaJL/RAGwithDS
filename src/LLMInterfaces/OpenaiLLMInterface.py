from LLMInterfaces.LLMSuperclass import LLMSuperclass
from config import OPENAI_GPT35_API_KEY
from openai import AzureOpenAI

class OpenaiLLMInterface(LLMSuperclass):
    endpoint = "https://manue-mftmnpg0-eastus2.cognitiveservices.azure.com/"
    model_name = "gpt-35-turbo"
    deployment = "gpt35-deployment"

    @staticmethod
    def isAvailable():
        # Azure openai doesn't have a method of testing this, other than making a chat call
        # which costs money. So just leave it.
        return True


    def respond(self, query: str, context_text: str) -> str:
        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=self.endpoint,
            api_key=OPENAI_GPT35_API_KEY,
        )

        promptWithContext = (self.promptStr +
        """
        Context:
        {context}
        """
        ).format(context=context_text)

        completion = client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": promptWithContext
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        raw_response = completion.choices[0].message.content
        if 'assistantfinal' in raw_response:    #Response contains analysis and reasoning, which we wanna skip
            return raw_response.split('assistantfinal', 1)[1].strip()
        else:
            return raw_response.strip()
