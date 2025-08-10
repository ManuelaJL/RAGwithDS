from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FileLoader import load_pdfs_from_folder
from config import FOLDER_PATH, EMBEDDING_MODEL, INDEX_NAME

print("Starting Embedder")
path = FOLDER_PATH
docs = load_pdfs_from_folder(path)

# Choose a small-but-powerful embedding model
# sentence-transformers/all-MiniLM-L6-v2") #This had trouble with german documents
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL) #pip install sentence-transformers

# Build a FAISS vector store from the documents (Facebook AI Similarity Search)
vectorstore = FAISS.from_documents(docs, embedding_model)

last_folder_name = path.split("\\")[-1]
vectorstore.save_local(INDEX_NAME)