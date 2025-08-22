from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FileLoader import load_pdfs_from_folder
from config import FOLDER_PATH, EMBEDDING_MODEL, INDEX_NAME
from typing import cast
import os
import gc

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) #pip install sentence-transformers
def getAllDocsFromIndex(): #Warning! This takes a lot of memory!

    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, INDEX_NAME)

    vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
    ))

    return vectorstore.similarity_search("placeholder query, just get everything", k=vectorstore.index.ntotal)

def getDocsFromIndexByFilter(folder: str, minpage = 0, maxpage = 99999):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, INDEX_NAME)

    vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
    ))

    filtered_docs = [doc for doc in vectorstore.docstore._dict.values() if
                     doc.metadata.get("source", "unknown").startswith(folder)
                     and doc.metadata.get("page") >= minpage
                     and doc.metadata.get("page") <= maxpage]

    del vectorstore  # Optional: release reference to full index
    gc.collect()

    return filtered_docs





if __name__ == "__main__" :
    print("Starting Embedder")
    path = FOLDER_PATH
    docs = load_pdfs_from_folder(path)

    # Build a FAISS vector store from the documents (Facebook AI Similarity Search)
    vectorstore = FAISS.from_documents(docs, embedding_model)

    last_folder_name = path.split("\\")[-1]
    vectorstore.save_local(INDEX_NAME)

    cache_path = INDEX_NAME + "/vector_cache_for_relevance_using_" + EMBEDDING_MODEL.replace("/", "_") + ".jsonl"
    if os.path.exists(cache_path):
        os.remove(cache_path)