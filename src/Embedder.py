from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FileLoader import load_pdfs_from_folder
from config import FOLDER_PATH, EMBEDDING_MODEL, INDEX_NAME
from typing import cast


embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) #pip install sentence-transformers
def getAllDocsFromIndex():

    vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
        INDEX_NAME,
        embedding_model,
        allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
    ))

    return vectorstore.similarity_search("placeholder query, just get everything", k=vectorstore.index.ntotal)



if __name__ == "__main__" :
    print("Starting Embedder")
    path = FOLDER_PATH
    docs = load_pdfs_from_folder(path)

    # Build a FAISS vector store from the documents (Facebook AI Similarity Search)
    vectorstore = FAISS.from_documents(docs, embedding_model)

    last_folder_name = path.split("\\")[-1]
    vectorstore.save_local(INDEX_NAME)