# https://ollama.com/download
# Installation automatically extracted files to: C:\Users\manue\AppData\Local\Programs\Ollama\
# To download the model: ollama pull mistral
# Run it on GPU: ollama run mistral
# Smaller: ollama pull mistral:Q4_K_M  Even smaller: ollama pull tinyllama
# To make it run on cpu instead: (and use powershell as administrator) type into terminal:
#   set OLLAMA_NUM_GPU_LAYERS=0
#   set CUDA_VISIBLE_DEVICES=
#   ollama run tinyllama
from langchain.chains.base import Chain
from langchain_community.llms import Ollama
from config import EMBEDDING_MODEL, INDEX_NAME

llm = Ollama(model="tinyllama")


from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import cast


# Load your saved FAISS index
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #This had trouble with german documents

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    INDEX_NAME,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))
retriever = vectorstore.as_retriever()      #gets the context
retriever.search_kwargs["k"] = 1
retriever.search_type = "mmr"

from langchain.prompts import PromptTemplate

template = """
You must begin by repeating the exact question word for word.
Use only the context below to answer the question. Do not add anything unrelated.

Question:
{question}

Context:
{context}

Answer:
"""


prompt = PromptTemplate.from_template(template)

qa_chain = cast(Chain, RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
))


print(f"retriever.search_type: {retriever.search_type}")
print(f"retriever.search_kwargs: {retriever.search_kwargs}" )

query = "Ziele der Regressionsanalyse"
#Beispielfrage: wovon ist die Welt durchdrungen? Answer P.61: Die Welt ist durchdrungen von Heterogenität
#Beispielfrage: Ziele der Regressionsanalyse? Antwort: P.5 has exactly that title
result = qa_chain.invoke({"query": query})

print("🔍 Answer:")
print(result["result"])

print("\n📚 Source pages:")
for doc in result["source_documents"]:
    print(f"{doc.metadata['filename']} — page {doc.metadata['page']}")

#======================================================================================================
# Troubleshooting why it isn't giving the right answer.
# I asked "Wovon ist die Welt durchdrungen, laut kausalanalyse.pdf?"
# Page 61 has the exact answer: "Die Welt ist durchdrungen von Heterogenität"

print("\n======================================checking similarity search\n")

docs = vectorstore.similarity_search_with_score(query, k=10)
for i, (doc, score) in enumerate(docs):
    page = doc.metadata.get('page_number')
    text = doc.page_content[:300]
    print(f"{i + 1:>3} Page: {page} | Score: {score:.4f} \nContent: {text}\n")
    if page == 61:
        print(f"Page 61 was ranked at position {i} out of {len(docs)}")


# Result: it contains the part I want:
# Page: 61
# Content: Der experimentelle Ansatz: Randomisierung
# Die Welt ist durchdrungen von Heterogenität und es ist nicht          <--
# möglich, alles ausser dem Treatment konstant zu halten.
# Der Einuss von beobachteten und unbeobachteten Störvariablen
# (confounders) kann jedoch durch zufällige Zuweisung des
# Treatments eliminiert w


# docs = vectorstore.similarity_search(query, k=300)
# for i, (doc, score) in enumerate(docs):
#     if doc.metadata.page("page_number") == 61:
#         print(f"Page 61 was ranked at position {i} out of {len(docs)}")
#         break

#yields: 🎯 Page 61 was ranked at position 129 out of 174

#If I use instread "was sagt das Dokument über Heterogenität, it ranks 8, which is still too low.



# print("\n=================================== RetrievalQA Source Documents:")
# for doc in result["source_documents"]:
#     print(f"{doc.metadata['page_number']} — {doc.page_content[:100]}")



#=======================================================================================================

# Conclusion: TinyLlama doesn't seem suited for RAG.


# --> The problem was I used a different embedder than when I made the index!