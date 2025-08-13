
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import json
from config import EMBEDDING_MODEL, INDEX_NAME
from Embedder import getAllDocsFromIndex

# Part of the chain that was done in Embedder.py
# parent document --> chunks --> vectorized into vectorstore (the index.faiss created in the other file is the vectorstore)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

documents = getAllDocsFromIndex()


import hashlib

def create_chunk_id(doc):
    input = f"{doc.metadata['file_path']}_{doc.metadata.get('page')}"
    hash_obj = hashlib.sha256(input.encode('utf-8'))
    return hash_obj.hexdigest()



if __name__ == "__main__":
    with open(INDEX_NAME + "/vector_cache_for_relevance_using_" + EMBEDDING_MODEL.replace("/", "_") + ".jsonl", "w") as f:
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