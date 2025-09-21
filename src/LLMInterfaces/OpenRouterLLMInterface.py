from LLMInterfaces.LLMSuperclass import LLMSuperclass
import requests
from configSecret import OPENROUTER_API_KEY
from openai import OpenAI

class OpenRouterLLMInterface(LLMSuperclass):
    model = "openai/gpt-oss-20b:free"  # https://openrouter.ai/openai/gpt-oss-20b:free

    @staticmethod
    def isAvailable():
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}"
        }

        response = requests.get("https://openrouter.ai/api/v1/models", headers=headers)

        if response.status_code == 200:
            models = response.json()
            model_ids = [model["id"] for model in models["data"]]

            target_model = OpenRouterLLMInterface.model
            if target_model in model_ids:
                print(f"✅ Model '{target_model}' is available.")
                return True
            else:
                print(f"❌ Model '{target_model}' is NOT available.")
                return False
        else:
            print(f"⚠️ Failed to fetch models. Status code: {response.status_code}")
            return False


    def respond(self, query: str, context_text: str) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        promptWithContext = (self.promptStr +
        """
        Context:
        {context}
        """
                             ).format(context=context_text)

        completion = client.chat.completions.create(
            model=OpenRouterLLMInterface.model,
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


# Beispielantwort auf "Was ist Regressionsanalyse":
# analysisThe user: "Was ist Regressionsanalyse?" They want answer using only context. The context: multiple bullet points about regression in German. They want a concise answer. According to context: Regressionsanalyse is a method for describing distribution of a variable conditioned on another. Also explains relationships, tests causal hypotheses. Directed relationships: dependent variable, independent variable. Also linear regression uses least squares, etc.
#
# They want article style? They didn't ask for meta. So answer in German likely. They gave German context. Our answer should come from context. Use only context. So:
#
# "Regressionsanalyse ist ein Mittel zur Deskription und Erklärung von Beziehungen zwischen Variablen. Es beschreibt die bedingte Verteilung einer abhängigen Variable Y in Abhängigkeit von unabhängigen Variablen X, testet kausale Hypothesen, liefert ein gerichtetes Verhältnis: eine Ergebnisvariable Y und eine oder mehrere erklärende Variablen X. Bei linearer Regression wird die Abhängigkeit mit einer linearen Funktion y = β1·x + β2 + ϵ modelliert und die Parameter mittels kleinster Quadrate geschätzt."
#
# Make sure not to add meta comments. The answer is derived entirely from context. They also mention bestm. So include that it's linear regression with least squares.
#
# Ok.assistantfinal**Regressionsanalyse** ist ein Verfahren, mit dem die Beziehung zwischen einer abhängigen Variable Y und einer oder mehreren unabhängigen Variablen X beschrieben, erklärt und quantifiziert wird.
#
# - Sie dient der **Deskription**: sie beschreibt die bedingte Verteilung von Y für verschiedene Werte von X.
# - Sie dient der **Erklärung**: sie testet kausale Hypothesen über Zusammenhänge und ermittelt die Wirkung von X auf Y.
# - Die Beziehungen sind **gerichtet**: es gibt eine Ergebnisvariable Y (abhängig) und eine oder mehrere erklärende Variablen X (unabhängig).
#
# Im konkreten Fall der **linearen Regresion** wird die Abhängigkeit mit einer Geraden modelliert:
#
# \[
#     y = \beta_1\,x + \beta_2 + \epsilon ,
# \]
#
# wobei \(\beta_1\) und \(\beta_2\) durch Schätzwerte \(b_1\) und \(b_2\) bestimmt werden. Die Gerade wird üblicherweise mit der Methode der kleinsten Quadrate angepasst.