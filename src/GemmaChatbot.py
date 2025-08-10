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

from langchain_community.llms import Ollama


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import cast
from config import EMBEDDING_MODEL, INDEX_NAME

def find_search_term(keyword: str, vectorstore, k=50):
    hits = vectorstore.similarity_search(keyword, k=k)
    pages = [ # = Set comprehension. Creates a set, which doesn't allow duplicates
        (doc.metadata.get('source'), doc.metadata.get('page_number'))
        # f"{hits[0].metadata.get('source')} P.{doc.metadata.get('page_number')}"
        for doc in hits
        if keyword.lower() in doc.page_content.lower()
    ]
    sorted_entries = sorted(pages, key=lambda x: (x[0], x[1]))
    return [f"{source} P.{page}" for source, page in sorted_entries]



from src.VectorCacheCreator import create_chunk_id

def meets_relevance_criteria(doc, query, vector_cache, min_relevance=0.75):
    query_embedding = embedding_model.embed_query(query)
    cache_chunk_ID = create_chunk_id(doc)
    doc_embedding_entry = vector_cache.get(cache_chunk_ID)
    if not doc_embedding_entry:
        print(f"Warning: could not verify relevance of document {doc.metadata['file_path']}_{doc.metadata.get('page')} (hash {cache_chunk_ID})")
        doc.metadata['score'] = 0
        return True
    doc_embedding = doc_embedding_entry['embedding']
    chunk_score = compute_score(doc_embedding, query_embedding)
    if debug:
        print(f"Score of {doc.metadata['file_path']}_{doc.metadata.get('page')} : {chunk_score}")
    doc.metadata['score'] = chunk_score
    return chunk_score >= min_relevance


from sklearn.metrics.pairwise import cosine_similarity

def compute_score(doc_embedding, query_embedding):
    return float(cosine_similarity([query_embedding], [doc_embedding])[0][0])


def removeNonPrintableCharacters(text): #Not used at the moment
    return ''.join(c for c in text if c.isprintable() or c in '\n\r')


debug = True
# Next steps:
# 4. Include more documents

# Part of the chain that was done in Embedder.py
# parent document --> chunks --> vectorized into vectorstore (the index.faiss created in the other file is the vectorstore)


embedding_model = HuggingFaceEmbeddings( #Pitfall! Use same model here!
    model_name=EMBEDDING_MODEL
)



vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    INDEX_NAME,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))


from langchain.prompts import PromptTemplate

template = """
You must begin by repeating the exact question word for word.
Then answer the question, using the context below. Do not add anything unrelated.
Avoid vague or general statements — draw from concrete details in the context. If the context does not contain an answer, say so clearly.

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
retriever.search_type = "similarity" #mmr



# qa_chain = cast(Chain, RetrievalQA.from_chain_type( #We no longer use this because we want to add a step inbetween retrieving and generating the answer
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True,
#     chain_type_kwargs={"prompt": prompt}
# ))

qa_chain_without_retriever = prompt | llm

# This is for checking if the found documents are relevant
import json
import os
vector_cache_name = INDEX_NAME + "/vector_cache_for_relevance_using_" + EMBEDDING_MODEL.replace("/", "_") + ".jsonl"
vector_cache = {}
if os.path.exists(vector_cache_name):
    with open(vector_cache_name, "r") as f:
        for line in f:
            entry = json.loads(line)
            vector_cache[entry["chunk_id"]] = entry
else:
    print("Warning: there is currently no vector cache, so I can't filter out less relevant documents. You may create one using VectorCacheCreator.py")
    print(f"The path I checked: {vector_cache_name}")



# Start the chat

while True:
    user_query = input("Ask a question, or type exit (or Search: term to search):\n")
    if(user_query.lower() in ["exit", "quit"]):
        print("Bye!")
        break
    if(user_query.lower().startswith("search:")):
        search_term = user_query[len("search:"):].strip()
        print("Searching for: [" + search_term + "]")
        pages = find_search_term(search_term, vectorstore)  #Will only find k results, depending on how much you set in the function
        print(f"Pages mentioning '{search_term}':\n",
              "\n".join(str(p) for p in pages))
    else:

        retrieved_docs = retriever.get_relevant_documents(user_query)

        if not retrieved_docs or all(len(doc.page_content.strip()) < 20 for doc in retrieved_docs):
            print("❌ No relevant pages found.")
            continue

        if vector_cache:    #Filtering out documents that aren't relevant enough
            filtered_docs = [doc for doc in retrieved_docs if meets_relevance_criteria(doc, user_query, vector_cache, 0.6)]
        else:
            filtered_docs = retrieved_docs

        if not filtered_docs:
            print("None of the retrieved pages were relevant enough to answer your query")
            continue

        filtered_docs.sort(key=lambda x: x.metadata['score'], reverse=True)

        context_text = "\n\n".join(doc.page_content for doc in filtered_docs)

        if not context_text.strip():
            print("⚠️ No usable context found. Skipping model invocation.")
            continue

        result = None
        try:
            result = qa_chain_without_retriever.invoke({
                "question": user_query,
                "context":  context_text
            })
        except Exception as e:
            print(f"Got an exception: {e}")
        if not result:
            print("Got no result")
        else:
            print("🔍 Answer:")
            print(result)

            print("\n📚 Source pages:")
            for doc in filtered_docs:
                clean_path = doc.metadata['file_path'].replace('\\\\', '\\')
                print(f"Page {doc.metadata['page']} of {clean_path}")
                print("\n\t" + doc.page_content.replace("\n", "\n\t") + "\n")
        print("\n\n")
