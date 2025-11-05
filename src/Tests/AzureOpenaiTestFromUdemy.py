# https://www.udemy.com/course/rag-with-azure-openai-ai-search-cosmosdb-graphrag-more/learn/lecture/46813833

import configSecret

service_endpoint = configSecret.AZURE_SEARCH_SERVICE_ENDPOINT
index_name = configSecret.AZURE_SEARCH_INDEX_NAME
key = configSecret.AZURE_SEARCH_API_KEY

from openai import AzureOpenAI

azure_openai_endpoint = configSecret.AZURE_OPENAI_ENDPOINT
azure_openai_key = configSecret.AZURE_OPENAI_KEY

azure_openai_client = AzureOpenAI(
    api_key=azure_openai_key,
    api_version="2024-02-15-preview",
    azure_endpoint=azure_openai_endpoint
)


def generate_embeddings(client, text):
    embedding_model = configSecret.EMBEDDING_ENGINE

    response = client.embeddings.create(
        input = text,
        model = embedding_model
    )

    embeddings=response.model_dump()
    return embeddings['data'][0]['embedding']

user_query = "Welche Fachhochschule wird erwähnt?"
vectorised_user_query = generate_embeddings(azure_openai_client, user_query) #returns something along the lines of [0.02322423, -0.00423423432 ...

context=[]

url = f"{service_endpoint}/indexes/{index_name}/docs/search?api-version=2023-11-01"

headers = {
    "Content-Type": "application/json",
    "api-key" : key
}

body = {
    "count" : True,
    "select": "chunk",
    "vectorQueries": [
        {
            "vector": vectorised_user_query,
            "k" : 3,
            "fields" : "text_vector",
            "kind" : "vector"
        }
    ]
}

import requests
import json

response = requests.post(url, headers=headers, data=json.dumps(body))
documents = response.json()['value']

for doc in documents:
    context.append(dict(
        {
            "chunk": doc['chunk'],
            "score": doc['@search.score']
        }
    ))

    for doc in context:
        print(doc)

system_prompt = f"""
You are meant to behave as a RAG chatbot that derives its context from a database of university slides stored in Azure AI Search Solution.
please answer strictly from the context of the database provided and if you don't have an answer please politely say no. Don't include any extra
information that is not in the context and don't include links as well.
the context passed to you will be in the form of a pythonic list with each object in the list having structure as follows:
"chunk": "the content of the slide"
"score": "the relevancy score of the review"

the pythonic list contains best 3 matches to the user query based on cosine similarity of the embeddings of the user query and the review descriptions.
please structure your answers ina very professional manner and in such a way that the user does not get to know that it's RAG working under the hood
and it's as if they are talking to a human.
"""

user_prompt = f""" the user query is: {user_query}
the context is: {context}
"""

chat_completion_response = azure_openai_client.chat.completions.create(
    model = configSecret.GPT_ENGINE,
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7
)

print(chat_completion_response.choices[0].message.content)

# Result: (for "Wie heisst die Fachhochschule")
# {'chunk': '#Methodengliederung\n(DA_1 S.12)#', 'score': 0.81734806}
# {'chunk': '#Methodengliederung\n(DA_1 S.12)#', 'score': 0.81734806}
# {'chunk': 'Berner Fachhochschule | CAS Datenanalyse | Zeitreihenanalyse | Kapitel 1 |  Prof. Dr. Raúl Gimeno 1\n\nGeben Sie für die folgenden Merkmale das jeweilige Skalenniveau und \nmögliche Merkmalausprägungen. Unterscheiden Sie die Merkmale ferner \nin diskrete und stetige und diskutieren Sie dabei Probleme der \nMessgenauigkeit.\n\nAuffrischung: Übung 1\n\nMerkmal Ausprägungen Skalenniveau Diskret?\na Gewicht\nb Akademischer Grad\nc Augenfarbe\nd Geschlecht\ne Nettoeinkommen in CHF\n\n\n\nBerner Fachhochschule | CAS Datenanalyse | Zeitreihenanalyse | Kapitel 1 |  Prof. Dr. Raúl Gimeno 2\n\n• Erstellen Sie einen Box-Plot für die Variable Autopreis.\n\n• Bestimmen Sie folgende Elemente des Box-Plots: \n\n1. Median \n\n2. erste und dritte Quartile und (Inter)Quartilabstand\n\n3. unterer und oberer Fühler (whisker)\n\n4. Gibt es Ausreisser ?\n\nAuffrischung: Übung 2\n\n\n\tAuffrischung: Übung 1\n\tAuffrischung: Übung 2', 'score': 0.8101638}
# {'chunk': '#Methodengliederung\n(DA_1 S.12)#', 'score': 0.81734806}
# {'chunk': 'Berner Fachhochschule | CAS Datenanalyse | Zeitreihenanalyse | Kapitel 1 |  Prof. Dr. Raúl Gimeno 1\n\nGeben Sie für die folgenden Merkmale das jeweilige Skalenniveau und \nmögliche Merkmalausprägungen. Unterscheiden Sie die Merkmale ferner \nin diskrete und stetige und diskutieren Sie dabei Probleme der \nMessgenauigkeit.\n\nAuffrischung: Übung 1\n\nMerkmal Ausprägungen Skalenniveau Diskret?\na Gewicht\nb Akademischer Grad\nc Augenfarbe\nd Geschlecht\ne Nettoeinkommen in CHF\n\n\n\nBerner Fachhochschule | CAS Datenanalyse | Zeitreihenanalyse | Kapitel 1 |  Prof. Dr. Raúl Gimeno 2\n\n• Erstellen Sie einen Box-Plot für die Variable Autopreis.\n\n• Bestimmen Sie folgende Elemente des Box-Plots: \n\n1. Median \n\n2. erste und dritte Quartile und (Inter)Quartilabstand\n\n3. unterer und oberer Fühler (whisker)\n\n4. Gibt es Ausreisser ?\n\nAuffrischung: Übung 2\n\n\n\tAuffrischung: Übung 1\n\tAuffrischung: Übung 2', 'score': 0.8101638}
# {'chunk': '#Spezialfall\nvon#\n          \n          \n            \n          \n          \n            \n          \n          \n            \n          \n          \n            \n          \n          \n            \n          \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n    \n    \n      \n        \n      \n      \n        \n      \n      \n        \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n        \n      \n    \n    \n      \n        \n      \n      \n        \n      \n      \n        \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n        \n      \n    \n    \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n      \n      \n        \n          \n            #2 Stichproben#', 'score': 0.808232}
# Die Fachhochschule, auf die sich die Folien beziehen, wird nicht direkt erwähnt. Leider kann ich daher nicht mit Sicherheit sagen, wie die Fachhochschule heißt.
# --> but it works for "Welche Fachhochschule wird erwähnt?"

# Result for "What is numpy"
# {'chunk': '9g6VSMJA3XoVvW2HZxfHWsA94pKkmNMNAlqREGuiQ1wuehX6FVb5zwE4ckTZgzdElqhIEuSY0w0CWpETfkOXTPg0tqkTN0SWqEgS5JjTDQJakRBrokNeKGvCiqBvjApuaM7NOOVvteafD7xBm6JDXCQJekRqzLUy6uI5ekZ3KGLkmNWJczdDXiBrpYpetbK7/1G+iSmjCyVTLrSKdAT7IL+F1gA/Dhqrp32fbvAP4IeAXwBPCzVfXYcEuVpOvPaj9IYPwz/IHn0JNsAI4AdwA7gLuS7Fg27K3Ak1X1/cD7gfcMu1BJ0tq6zNB3AvNVdQ4gyTFgD3C6b8we4FDv9QPAB5KkqmqItUrSeK2z6zwZlLlJ3gDsqqq39dpvBl5ZVfv7xny+N2ah1/733pgvL3uvfcC+XvPFwNlhHcgE3Ap8eeCo9aXFY4I2j8tjWj+GfVwvrKqplTaM9aJoVR0Fjo5zn6OSZK6qpiddxzC1eEzQ5nF5TOvHOI+ryzr0C8CWvvbmXt+KY5LcBNzC0sVRSdKYdAn0E8D2JNuSbAT2AjPLxswAb+m9fgPwSc+fS9J4DTzlUlWXkuwHjrO0bPG+qjqV5DAwV1UzwO8Df5xkHvgKS6HfuiZOHS3T4jFBm8flMa0fYzuugRdFJUnrg89ykaRGGOiS1AgD/SoleW+SLyR5NMnHkzxv0jUNQ5I3JjmV5FtJ1vUSsiS7kpxNMp/kwKTrGYYk9yV5vHfvRxOSbEnycJLTve+9d0y6pmuV5OYk/5TkX3rH9Fvj2K+BfvUeAl5SVS8FvgjcM+F6huXzwE8Dn5p0Idei4yMr1qM/AHZNuoghuwS8q6p2AK8C3t7Av9XXgddW1Q8CLwN2JXnVqHdqoF+lqvqbqrrUaz7C0vr8da+qzlTVer6D97KnH1lRVU8Blx9Zsa5V1adYWknWjKr6UlV9rvf6v4EzwKbJVnVtasn/9JrP7n2NfAWKgT4cPw98YtJF6NtsAs73tRdY5yFxI0iyFXg58JnJVnLtkmxIchJ4HHioqkZ+TD4PfQ1J/hb43hU2vbuq/rI35t0s/cr4kXHWdi26HJc0bkmeA/wZ8CtV9bVJ13OtquqbwMt619c+nuQlVTXSax8G+hqq6ra1tie5G/hJ4MfX052xg46rEV0eWaHrRJJnsxTmH6mqP590PcNUVV9N8jBL1z5GGuiecrlKvQ/9+HVgd1X976Tr0TN0eWSFrgNJwtLd5meq6n2TrmcYkkxdXvmW5DuB24EvjHq/BvrV+wDwXOChJCeTfHDSBQ1Dkp9KsgD8CPBgkuOTrulq9C5YX35kxRng/qo6Ndmqrl2SjwGfBl6cZCHJWydd0xC8Gngz8Nre/6WTSX5i0kVdo+cDDyd5lKXJxUNV9dej3qm3/ktSI5yhS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUiP8HO+ALwCQnn4IAAAAASUVORK5CYII=\\n",\n      "text/plain": [\n       "<Figure size 432x288 with 1 Axes>"\n      ]\n     },\n     "metadata": {\n      "needs_background": "light"\n     },\n     "output_type": "display_data"\n    }\n   ],\n   "source": [\n    "norm_dist1 = np.random.normal(0, 1, 1000)\\n",\n    "norm_dist2 = np.random.normal(1, 0.5, 1000)\\n",\n    "plt.hist([norm_dist1,norm_dist2], bins=20, density=True)\\n",\n    "plt.show()\\n",\n    "plt.close()"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [', 'score': 0.8021449}
# {'chunk': '9g6VSMJA3XoVvW2HZxfHWsA94pKkmNMNAlqREGuiQ1wuehX6FVb5zwE4ckTZgzdElqhIEuSY0w0CWpETfkOXTPg0tqkTN0SWqEgS5JjTDQJakRBrokNeKGvCiqBvjApuaM7NOOVvteafD7xBm6JDXCQJekRqzLUy6uI5ekZ3KGLkmNWJczdDXiBrpYpetbK7/1G+iSmjCyVTLrSKdAT7IL+F1gA/Dhqrp32fbvAP4IeAXwBPCzVfXYcEuVpOvPaj9IYPwz/IHn0JNsAI4AdwA7gLuS7Fg27K3Ak1X1/cD7gfcMu1BJ0tq6zNB3AvNVdQ4gyTFgD3C6b8we4FDv9QPAB5KkqmqItUrSeK2z6zwZlLlJ3gDsqqq39dpvBl5ZVfv7xny+N2ah1/733pgvL3uvfcC+XvPFwNlhHcgE3Ap8eeCo9aXFY4I2j8tjWj+GfVwvrKqplTaM9aJoVR0Fjo5zn6OSZK6qpiddxzC1eEzQ5nF5TOvHOI+ryzr0C8CWvvbmXt+KY5LcBNzC0sVRSdKYdAn0E8D2JNuSbAT2AjPLxswAb+m9fgPwSc+fS9J4DTzlUlWXkuwHjrO0bPG+qjqV5DAwV1UzwO8Df5xkHvgKS6HfuiZOHS3T4jFBm8flMa0fYzuugRdFJUnrg89ykaRGGOiS1AgD/SoleW+SLyR5NMnHkzxv0jUNQ5I3JjmV5FtJ1vUSsiS7kpxNMp/kwKTrGYYk9yV5vHfvRxOSbEnycJLTve+9d0y6pmuV5OYk/5TkX3rH9Fvj2K+BfvUeAl5SVS8FvgjcM+F6huXzwE8Dn5p0Idei4yMr1qM/AHZNuoghuwS8q6p2AK8C3t7Av9XXgddW1Q8CLwN2JXnVqHdqoF+lqvqbqrrUaz7C0vr8da+qzlTVer6D97KnH1lRVU8Blx9Zsa5V1adYWknWjKr6UlV9rvf6v4EzwKbJVnVtasn/9JrP7n2NfAWKgT4cPw98YtJF6NtsAs73tRdY5yFxI0iyFXg58JnJVnLtkmxIchJ4HHioqkZ+TD4PfQ1J/hb43hU2vbuq/rI35t0s/cr4kXHWdi26HJc0bkmeA/wZ8CtV9bVJ13OtquqbwMt619c+nuQlVTXSax8G+hqq6ra1tie5G/hJ4MfX052xg46rEV0eWaHrRJJnsxTmH6mqP590PcNUVV9N8jBL1z5GGuiecrlKvQ/9+HVgd1X976Tr0TN0eWSFrgNJwtLd5meq6n2TrmcYkkxdXvmW5DuB24EvjHq/BvrV+wDwXOChJCeTfHDSBQ1Dkp9KsgD8CPBgkuOTrulq9C5YX35kxRng/qo6Ndmqrl2SjwGfBl6cZCHJWydd0xC8Gngz8Nre/6WTSX5i0kVdo+cDDyd5lKXJxUNV9dej3qm3/ktSI5yhS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUiP8HO+ALwCQnn4IAAAAASUVORK5CYII=\\n",\n      "text/plain": [\n       "<Figure size 432x288 with 1 Axes>"\n      ]\n     },\n     "metadata": {\n      "needs_background": "light"\n     },\n     "output_type": "display_data"\n    }\n   ],\n   "source": [\n    "norm_dist1 = np.random.normal(0, 1, 1000)\\n",\n    "norm_dist2 = np.random.normal(1, 0.5, 1000)\\n",\n    "plt.hist([norm_dist1,norm_dist2], bins=20, density=True)\\n",\n    "plt.show()\\n",\n    "plt.close()"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [', 'score': 0.8021449}
# {'chunk': 'axis=0).T\\n",\r\n    "# nfq_min_t, nfq_min_r, nfq_min_s, \\\\\\n",\r\n    "#     nfq_min_sec, nfq_min_rt = np.nanmin(nfq_results, axis=0).T\\n",\r\n    "# nfq_mean_t, nfq_mean_r, nfq_mean_s, \\\\\\n",\r\n    "#     nfq_mean_sec, nfq_mean_rt = np.nanmean(nfq_results, axis=0).T\\n",\r\n    "# nfq_x = np.arange(len(nfq_mean_s))"\r\n   ]\r\n  },\r\n  {\r\n   "cell_type": "code",\r\n   "execution_count": 18,\r\n   "metadata": {\r\n    "scrolled": false\r\n   },\r\n   "outputs": [\r\n    {\r\n     "data": {\r\n      "image/png": "iVBORw0KGgoAAAANSUhEUgAABAoAAAcyCAYAAAAQZQdxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAgAElEQVR4nOzdd3hUVfrA8e/0SZ/0kISqgiIoKqJSBJQiiIgURVGxoeK6q6L+rGtZdWXXrrisBRsK0kTEVUFUVBARkS49JKSQhPQyfeb+/hjmmklmkkAq+H6eh8fk3nPvPXeSGXPe+573aMrKyhSEEEIIIYQQQgghAG1bd0AIIYQQQgghhBDthwQKhBBCCCGEEEIIoZJAgRBCCCGEEEIIIVQSKBBCCCGEEEIIIYRKAgVCCCGEEEIIIYRQSaBACCGEEEIIIYQQKgkUCCGEaFbTp0/HYrHQu3fvtu6KEMe9d955B4vFgsVioaCgoMnn++ijj7BYLPTt2xePx9MMPWx799xzDxaLhe7du7fYNQoKCtSfwzvvvNNi12kOubm5JCcnk5CQwK5du9q6O0KI45QECoQQopX8+OOP6h+aFouFlJQUysvLG3XsOeecE3Bse/9D9UT35ptvqj+LhIQECgsL27pLzW716tUBv3O1f3d79uzJpEmTmDNnDtXV1W3dXdEIVVVV/OMf/wDg/vvvR6fTATBs2LCQP+vG/nvppZfa8tZEDWlpaVx77bW43W4efvjhtu6OEOI4JYECIYRoI3a7nWXLljXYbv369ezfv78VeiQaa/78+erXbrebRYsWtWFvWp/dbicvL4+vv/6ae++9l/79+7Njx4627pZowOzZsykoKODkk09m4sSJbd0d0YLuuecejEYj3377LT/++GNbd0cIcRzSt3UHhBDiz8hsNmO32/n444+5/vrr62378ccfAxAWFobNZmuN7jXJ7NmzmT17dlt3o8Xs2rWLTZs2ARAZGUlVVRXz58/nL3/5Sxv3rOXceuut3HDDDer3xcXF7N27l1mzZpGRkUFWVhZXXnkl69evJzIysu06KkKqrq7m9ddfB3zTg/zZBABvv/02Vqs16HGzZ89m7ty5ALz11lucfvrpQdslJyc3c48b76WXXmrxjIbk5GTKyspa9BrNKT09nXHjxrFw4UL+/e9/M2jQoLbukhDiOCMZBUII0QZGjx4NwLp168jKygrZzuFwsHTp0oBjRNvyZxOEhYXxxBNPALB9+3a2bdvWhr1qWYmJifTs2VP9N2jQIG666SZ+/vlnBgwYAPjmRfsHlKL9mT9/PmVlZRgMBq644oqAfV26dAn4+db8Fx8ff9TtRPswadIkwDftbfv27W3cGyHE8UYCBUII0QYGDBhAx44dURSFhQsXhmz35ZdfUlZWhslkYty4ca3YQxGM1+tVf16jR4/m6quvJiIiAgicjvBnYTQaeeCBB9TvV69e3XadEfV6//33AV89gr', 'score': 0.8015248}
# {'chunk': '9g6VSMJA3XoVvW2HZxfHWsA94pKkmNMNAlqREGuiQ1wuehX6FVb5zwE4ckTZgzdElqhIEuSY0w0CWpETfkOXTPg0tqkTN0SWqEgS5JjTDQJakRBrokNeKGvCiqBvjApuaM7NOOVvteafD7xBm6JDXCQJekRqzLUy6uI5ekZ3KGLkmNWJczdDXiBrpYpetbK7/1G+iSmjCyVTLrSKdAT7IL+F1gA/Dhqrp32fbvAP4IeAXwBPCzVfXYcEuVpOvPaj9IYPwz/IHn0JNsAI4AdwA7gLuS7Fg27K3Ak1X1/cD7gfcMu1BJ0tq6zNB3AvNVdQ4gyTFgD3C6b8we4FDv9QPAB5KkqmqItUrSeK2z6zwZlLlJ3gDsqqq39dpvBl5ZVfv7xny+N2ah1/733pgvL3uvfcC+XvPFwNlhHcgE3Ap8eeCo9aXFY4I2j8tjWj+GfVwvrKqplTaM9aJoVR0Fjo5zn6OSZK6qpiddxzC1eEzQ5nF5TOvHOI+ryzr0C8CWvvbmXt+KY5LcBNzC0sVRSdKYdAn0E8D2JNuSbAT2AjPLxswAb+m9fgPwSc+fS9J4DTzlUlWXkuwHjrO0bPG+qjqV5DAwV1UzwO8Df5xkHvgKS6HfuiZOHS3T4jFBm8flMa0fYzuugRdFJUnrg89ykaRGGOiS1AgD/SoleW+SLyR5NMnHkzxv0jUNQ5I3JjmV5FtJ1vUSsiS7kpxNMp/kwKTrGYYk9yV5vHfvRxOSbEnycJLTve+9d0y6pmuV5OYk/5TkX3rH9Fvj2K+BfvUeAl5SVS8FvgjcM+F6huXzwE8Dn5p0Idei4yMr1qM/AHZNuoghuwS8q6p2AK8C3t7Av9XXgddW1Q8CLwN2JXnVqHdqoF+lqvqbqrrUaz7C0vr8da+qzlTVer6D97KnH1lRVU8Blx9Zsa5V1adYWknWjKr6UlV9rvf6v4EzwKbJVnVtasn/9JrP7n2NfAWKgT4cPw98YtJF6NtsAs73tRdY5yFxI0iyFXg58JnJVnLtkmxIchJ4HHioqkZ+TD4PfQ1J/hb43hU2vbuq/rI35t0s/cr4kXHWdi26HJc0bkmeA/wZ8CtV9bVJ13OtquqbwMt619c+nuQlVTXSax8G+hqq6ra1tie5G/hJ4MfX052xg46rEV0eWaHrRJJnsxTmH6mqP590PcNUVV9N8jBL1z5GGuiecrlKvQ/9+HVgd1X976Tr0TN0eWSFrgNJwtLd5meq6n2TrmcYkkxdXvmW5DuB24EvjHq/BvrV+wDwXOChJCeTfHDSBQ1Dkp9KsgD8CPBgkuOTrulq9C5YX35kxRng/qo6Ndmqrl2SjwGfBl6cZCHJWydd0xC8Gngz8Nre/6WTSX5i0kVdo+cDDyd5lKXJxUNV9dej3qm3/ktSI5yhS1IjDHRJaoSBLkmNMNAlqREGuiQ1wkCXpEYY6JLUiP8HO+ALwCQnn4IAAAAASUVORK5CYII=\\n",\n      "text/plain": [\n       "<Figure size 432x288 with 1 Axes>"\n      ]\n     },\n     "metadata": {\n      "needs_background": "light"\n     },\n     "output_type": "display_data"\n    }\n   ],\n   "source": [\n    "norm_dist1 = np.random.normal(0, 1, 1000)\\n",\n    "norm_dist2 = np.random.normal(1, 0.5, 1000)\\n",\n    "plt.hist([norm_dist1,norm_dist2], bins=20, density=True)\\n",\n    "plt.show()\\n",\n    "plt.close()"\n   ]\n  },\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [', 'score': 0.8021449}
# {'chunk': 'axis=0).T\\n",\r\n    "# nfq_min_t, nfq_min_r, nfq_min_s, \\\\\\n",\r\n    "#     nfq_min_sec, nfq_min_rt = np.nanmin(nfq_results, axis=0).T\\n",\r\n    "# nfq_mean_t, nfq_mean_r, nfq_mean_s, \\\\\\n",\r\n    "#     nfq_mean_sec, nfq_mean_rt = np.nanmean(nfq_results, axis=0).T\\n",\r\n    "# nfq_x = np.arange(len(nfq_mean_s))"\r\n   ]\r\n  },\r\n  {\r\n   "cell_type": "code",\r\n   "execution_count": 18,\r\n   "metadata": {\r\n    "scrolled": false\r\n   },\r\n   "outputs": [\r\n    {\r\n     "data": {\r\n      "image/png": "iVBORw0KGgoAAAANSUhEUgAABAoAAAcyCAYAAAAQZQdxAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8li6FKAAAgAElEQVR4nOzdd3hUVfrA8e/0SZ/0kISqgiIoKqJSBJQiiIgURVGxoeK6q6L+rGtZdWXXrrisBRsK0kTEVUFUVBARkS49JKSQhPQyfeb+/hjmmklmkkAq+H6eh8fk3nPvPXeSGXPe+573aMrKyhSEEEIIIYQQQgghAG1bd0AIIYQQQgghhBDthwQKhBBCCCGEEEIIoZJAgRBCCCGEEEIIIVQSKBBCCCGEEEIIIYRKAgVCCCGEEEIIIYRQSaBACCGEEEIIIYQQKgkUCCGEaFbTp0/HYrHQu3fvtu6KEMe9d955B4vFgsVioaCgoMnn++ijj7BYLPTt2xePx9MMPWx799xzDxaLhe7du7fYNQoKCtSfwzvvvNNi12kOubm5JCcnk5CQwK5du9q6O0KI45QECoQQopX8+OOP6h+aFouFlJQUysvLG3XsOeecE3Bse/9D9UT35ptvqj+LhIQECgsL27pLzW716tUBv3O1f3d79uzJpEmTmDNnDtXV1W3dXdEIVVVV/OMf/wDg/vvvR6fTATBs2LCQP+vG/nvppZfa8tZEDWlpaVx77bW43W4efvjhtu6OEOI4JYECIYRoI3a7nWXLljXYbv369ezfv78VeiQaa/78+erXbrebRYsWtWFvWp/dbicvL4+vv/6ae++9l/79+7Njx4627pZowOzZsykoKODkk09m4sSJbd0d0YLuuecejEYj3377LT/++GNbd0cIcRzSt3UHhBDiz8hsNmO32/n444+5/vrr62378ccfAxAWFobNZmuN7jXJ7NmzmT17dlt3o8Xs2rWLTZs2ARAZGUlVVRXz58/nL3/5Sxv3rOXceuut3HDDDer3xcXF7N27l1mzZpGRkUFWVhZXXnkl69evJzIysu06KkKqrq7m9ddfB3zTg/zZBABvv/02Vqs16HGzZ89m7ty5ALz11lucfvrpQdslJyc3c48b76WXXmrxjIbk5GTKyspa9BrNKT09nXHjxrFw4UL+/e9/M2jQoLbukhDiOCMZBUII0QZGjx4NwLp168jKygrZzuFwsHTp0oBjRNvyZxOEhYXxxBNPALB9+3a2bdvWhr1qWYmJifTs2VP9N2jQIG666SZ+/vlnBgwYAPjmRfsHlKL9mT9/PmVlZRgMBq644oqAfV26dAn4+db8Fx8ff9TtRPswadIkwDftbfv27W3cGyHE8UYCBUII0QYGDBhAx44dURSFhQsXhmz35ZdfUlZWhslkYty4ca3YQxGM1+tVf16jR4/m6quvJiIiAgicjvBnYTQaeeCBB9TvV69e3XadEfV6//33AV89gr', 'score': 0.8015248}
# {'chunk': 'Data Mining\n\nAndrea Giovannini\n\nLecture 1.5\n\nData frames and data \nmanipulation\n\n\n\nIn this lecture, we will look at\n\n• The Data Frame Object\n• Manipulating Data\n\n\n\nThe Data Frame object\n\n• Data Frame is a “Matrix” with mixed column types\n• Any operation on a Data Frame also work on matrices\n• Mostly the first object you obtain when loading data into R\n• Data Frames often are cohered to matrices when needed (manually / automatic)\n• Keeps track on meta data (e.g. column / row names, column types, stats)\n• Very easy to transform, filter and rearrange data\n\n\n\nThe Data Frame object\n\n\n\nManipulating Data\n\n• Filtering columns and rows is covered on Section 3\n• Here we concentrate in actually changing data within the data frame\n• Functional programming is very handy when changing a lot of data\n\n\n\nExample: Data Normalizing\n\n• Normalizing data is essential to many data mining algorithms\n• House prizes are from 100 K to 1000 K\n• Number of rooms are from 1 to 10\n• Normalizing squashes all values to the range from 0 to 1\n• Preserves distribution\n\n• Let’s see how to do this in R by building a function\n\n\n\n\n\nSummary\n\nConclusion: R is a very powerful open source domain specific language for statistical\nanalysis with nearly unlimited possibilities brought by the language itself and more\nthan 8000 additional free packages\n\n\n\nNext Section\n\nClustering – find similarities within your data', 'score': 0.80101556}
# Numpy is a fundamental package for scientific computing in Python. It is used for working with arrays, linear algebra, Fourier transforms, and random number capabilities.

