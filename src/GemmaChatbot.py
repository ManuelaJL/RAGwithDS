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
# Tried instead: Ollama pull gemma:2b, and that worked
from langchain.chains.base import Chain
from langchain_community.llms import Ollama


from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import cast

# Part of the chain that was done in Embedder.py
# parent document --> chunks --> vectorized into vectorstore (the index.faiss created in the other file is the vectorstore)
indexName = "my_index_of_Wahltag Kausalanalyse"

# Load your saved FAISS index
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #This had trouble with german documents

embedding_model = HuggingFaceEmbeddings( #Pitfall! Use same model here!
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)



vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    indexName,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))


from langchain.prompts import PromptTemplate

template = """
You must begin by repeating the exact question word for word.
Then answer the question, using only the context below to answer the question. Do not add anything unrelated.
Avoid vague or general statements — provide a technically accurate response.

Question:
{question}

Context:
{context}

Answer:
"""


prompt = PromptTemplate.from_template(template)

llm = Ollama(model="gemma:2b") #LangChain LLM wrapper

retriever = vectorstore.as_retriever()      #gets the context
# retriever.search_kwargs["k"] = 1
# retriever.search_type = "mmr"



qa_chain = cast(Chain, RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
))


# print(f"retriever.search_type: {retriever.search_type}")
# print(f"retriever.search_kwargs: {retriever.search_kwargs}" )

query = "Was sind die Ziele der Regressionsanalyse?"
#Beispielfrage: wovon ist die Welt durchdrungen? Answer P.61: Die Welt ist durchdrungen von Heterogenität
#Beispielfrage: Ziele der Regressionsanalyse? Antwort: P.5 has exactly that title

result_from_sequence = qa_chain.invoke({"query": query}) #Comment this out when using the below troubleshooting


print("🔍 Answer:")
print(result_from_sequence["result"])

print("\n📚 Source pages:")
for doc in result_from_sequence["source_documents"]:
    print(f"{doc.metadata['filename']} — page {doc.metadata['page']}")



#============ Troubleshooting: forcing the right page into the context, ignoring the rest ===============

# target_doc = [
#     doc for doc in vectorstore.similarity_search("Ziele der Regressionsanalyse", k=300)
#     if doc.metadata.get("page_number") == 5
# ][0]
#
#
# qa_chain_without_retriever = prompt | llm
#
# result = qa_chain_without_retriever.invoke({
#     "question": "Was sind die Ziele der Regressionsanalyse?",
#     "context": target_doc.page_content
# })
# print("=======================Test run=====================")
# print("🔍 Answer:")
# print(result)
# print("=======================End of test run=====================")

# # Result:
# # Reminder: content of page is
# # Ziele der Regressionsanalyse
# # Regression als ein Mittel zur Deskription
# # I Beschreibung der konditionalen Verteilung einer Variablen.
# # I Wie unterscheidet sich die Verteilung von Y für verschiedene Werte
# # von X?
# # Regression als ein Mittel zur Erklärung
# # I Test von (kausalen) Hypothesen über Zusammenhänge zwischen
# # Variablen
# # I Wie ist der Eekt von X auf Y ?
# # Gerichtete Beziehungen
# # I Anders als in der Korrelationsanalyse sind in der Regressionsanalyse
# # die Beziehungen zwischen den Variablen gerichtet
# # F Es gibt eine abhängige Variable (erklärte Variable, Ergebnisvariable,
# #                                             Response, Regressand, . . . )
# # F und es gibt unabhängige Variablen (erklärende Variable, Kovariate,
# #                                                 Stimulus, Regressor, . . . ).
# #
# # Prompt: You must begin by repeating the exact question word for word.
# #     Then answer the question, using only the context below to answer the question. Do not add anything unrelated.
# #
# #
# # And what I got as a result:
# # 🔍 Answer:
# # **Was sind die Ziele der Regressionsanalyse?**
# #
# # Ziele der Regressionsanalyse sind die Bestimmung der Muster und Regeln, die die Verteilung der Variablen beeinflussen.
# #
# # The answer isn't always the same! There's randomness to it! (temperature)

#========================================================================================================