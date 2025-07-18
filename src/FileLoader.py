


import os

def getContentsOfSinglePDFFile(path: str):
    from langchain_community.document_loaders import PyMuPDFLoader
    path = os.path.normpath(path)  # makes sure slashes are treated correctly (i.e. / turned into \ etc.)
    try:
        loader = PyMuPDFLoader(path)
        content = loader.load()
        return content
    except Exception as e:
        print(f"Failed to load: {e}")

def load_pdfs_from_folder(path: str): #This will make one document per page
    all_docs = []
    path = os.path.normpath(path)  # makes sure slashes are treated correctly

    for filename in os.listdir(path):
        if(filename.lower().endswith(".pdf")):
            full_path = os.path.join(path, filename)
            docs = getContentsOfSinglePDFFile(full_path)

            if docs:  # avoid crashing if file failed to load
                for page_number, doc in enumerate(docs, start=1):
                    doc.metadata["filename"] = filename
                    doc.metadata["page_number"] = page_number  # 🔢 Add page number metadata

                all_docs.extend(docs)


    print(f"Total documents loaded: {len(all_docs)}")
    #Metadata they have after this: source (file path), file_path (same thing), page, total_pages, format: PDF, title, author, filename, and a bunch of other useless stuff
    return all_docs

if __name__ == "__main__":
    print("starting FileLoader")
    path = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\Wahltag Kausalanalyse' #\kausalanalyse.pdf
    docs = load_pdfs_from_folder(path)

    for doc in docs:
        print(f"====================={doc.metadata['filename']} Page {doc.metadata['page']}=============================")
        print(doc)

    print("End of FileLoader")