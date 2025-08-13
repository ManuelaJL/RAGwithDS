
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from typing import cast
from config import EMBEDDING_MODEL, INDEX_NAME, FOLDER_PATH

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    INDEX_NAME,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))

documents = vectorstore.similarity_search("placeholder query, just get everything", k=vectorstore.index.ntotal)

set_of_sources = set()

for doc in documents:
    set_of_sources.add(doc.metadata.get("source", "unknown"))

sorted_list = sorted(set_of_sources)

print("List of sources that appear in " + INDEX_NAME)
print("\n".join(sorted_list))

print("Longest source:")
longest = max(sorted_list, key=len)
print(longest)

import os
print("Basepath: ", FOLDER_PATH)
for filename in os.listdir(FOLDER_PATH):
    full_path = os.path.join(FOLDER_PATH, filename)
    if os.path.isdir(full_path):
        print("  └─", filename)
