

from config import INDEX_NAME, FOLDER_PATH

from Embedder import getAllDocsFromIndex

documents = getAllDocsFromIndex()

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
