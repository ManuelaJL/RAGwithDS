
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import cast
from tqdm import tqdm
import json

# Part of the chain that was done in Embedder.py
# parent document --> chunks --> vectorized into vectorstore (the index.faiss created in the other file is the vectorstore)
indexName = "my_index_of_Wahltag Kausalanalyse"

embedModelString = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedding_model = HuggingFaceEmbeddings( #Pitfall! Use same model here!
    model_name=embedModelString
)

vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    indexName,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))

documents = vectorstore.similarity_search("placeholder query, just get everything", k=vectorstore.index.ntotal)


import hashlib

def create_chunk_id(doc):
    input = f"{doc.metadata['file_path']}_{doc.metadata.get('page')}"
    hash_obj = hashlib.sha256(input.encode('utf-8'))
    return hash_obj.hexdigest()



if __name__ == "__main__":
    with open(indexName + "/vector_cache_for_relevance_using_" + embedModelString.replace("/", "_") + ".jsonl", "w") as f:
        print("Creating vector cache")
        for doc in tqdm(documents):  #tqdm adds a progress bar
            embedding = embedding_model.embed_documents([doc.page_content])[0]

            cache_entry = {
                "chunk_id": create_chunk_id(doc),
                "embedding": embedding,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None)
            }

            f.write(json.dumps(cache_entry) + "\n")