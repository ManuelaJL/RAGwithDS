
import os
from config import FOLDER_PATH

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
        full_path = os.path.join(path, filename)
        if os.path.isdir(full_path):
            print(f"Entering folder {full_path}")
            foldercontents = load_pdfs_from_folder(full_path)
            all_docs.extend(foldercontents)
        elif filename.lower().endswith(".pdf"):
            full_path = os.path.join(path, filename)
            docs = getContentsOfSinglePDFFile(full_path)

            if docs:  # avoid crashing if file failed to load
                for page_number, doc in enumerate(docs, start=1):
                    doc.metadata["filename"] = filename
                    doc.metadata["page_number"] = page_number

                all_docs.extend(docs)


    print(f"Total documents loaded: {len(all_docs)}")
    #Metadata they have after this: source (file path), file_path (same thing), page, total_pages, format: PDF, title, author, filename, and a bunch of other useless stuff
    return all_docs

if __name__ == "__main__":
    print("starting FileLoader")
    docs = load_pdfs_from_folder(FOLDER_PATH)

    for doc in docs:
        print(f"====================={doc.metadata['filename']} Page {doc.metadata['page']}=============================")
        print(doc)

    print("End of FileLoader")