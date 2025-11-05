# https://platform.openai.com/docs/overview
# https://platform.openai.com/usage
# Copy the API-key they give you
# pip install openai
# Incase I need this later: Api key name: manusAPIKey

from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)

response = client.responses.create(
    model="gpt-3.5-turbo",
    input="write a haiku about ai",
    store=True,
)

print(response.output_text)

# To connection OpenAI models to Azure:
# https://platform.openai.com/account/api-keys -> get secret key
# https://portal.azure.com -> Click on one of your search services, e.g. mas-rag-aisearch-eastus
# -> import data (new) ->