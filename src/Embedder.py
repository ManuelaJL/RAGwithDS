from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from FileLoader import load_pdfs_from_folder

print("Starting Embedder")
path = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\Wahltag Kausalanalyse' #\kausalanalyse.pdf
docs = load_pdfs_from_folder(path)

# Choose a small-but-powerful embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #pip install sentence-transformers

# Build a FAISS vector store from the documents (Facebook AI Similarity Search)
vectorstore = FAISS.from_documents(docs, embedding_model)

last_folder_name = path.split("\\")[-1]
vectorstore.save_local(f"my_index_of_{last_folder_name}")

# retriever = vectorstore.as_retriever(search_type="similarity", k=4) #Searches for chunks relevant to the question
#
# from langchain.chains import RetrievalQA
# from langchain_community.llms import HuggingFaceHub  # or other model client
#
# llm = HuggingFaceHub(repo_id="google/flan-t5-base")  # You can use others like OpenAI if you have keys
#
# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     return_source_documents=True
# )
#
# query = "What is the difference between correlation and causation?"
# result = qa_chain({"query": query})
#
# print("🔍 Answer:")
# print(result["result"])
#
# print("\n📚 Source pages:")
# for doc in result["source_documents"]:
#     print(f"{doc.metadata['filename']} — page {doc.metadata.get('page_number')}")