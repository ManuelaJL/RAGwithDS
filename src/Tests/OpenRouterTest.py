# https://openrouter.ai/docs/quickstart
# https://openrouter.ai/docs/api-reference/list-available-models
# https://openrouter.ai/models

from openai import OpenAI
from configSecret import OPENROUTER_API_KEY     # not in git history

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

completion = client.chat.completions.create(
    # extra_headers={
    #     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    #     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    # },
    model="openai/gpt-oss-20b:free", # https://openrouter.ai/openai/gpt-oss-20b:free
    messages=[
        {
            "role": "system",
            "content": "You're a helpful assistant who speaks like a pirate."
        },
        {
            "role": "user",
            "content": "What is the meaning of life?"
        }
    ]
)

print(completion.choices[0].message.content)
