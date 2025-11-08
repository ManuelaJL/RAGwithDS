


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from typing import cast

from LLMInterfaces.OpenRouterLLMInterface import OpenRouterLLMInterface
from LLMInterfaces.TinyOllamaLLMInterface import TinyOllamaLLMInterface
from LLMInterfaces.AzureOpenaiLLMInterface import AzureOpenaiLLMInterface
from LLMInterfaces.OllamaLLMInterface import OllamaLLMInterface
from config import EMBEDDING_MODEL, INDEX_NAME

def find_search_term(keyword: str, vectorstore, k=50):
    hits = vectorstore.similarity_search(keyword, k=k)
    pages = [ # = Set comprehension. Creates a set, which doesn't allow duplicates
        (doc.metadata.get('source'), doc.metadata.get('page_number'))
        # f"{hits[0].metadata.get('source')} P.{doc.metadata.get('page_number')}"
        for doc in hits
        if keyword.lower() in doc.page_content.lower()
    ]
    sorted_entries = sorted(pages, key=lambda x: (x[0], x[1]))
    return [f"{source} P.{page}" for source, page in sorted_entries]



from src.VectorCacheCreator import create_chunk_id

def meets_relevance_criteria(doc, query, vector_cache, min_relevance=0.75):
    query_embedding = embedding_model.embed_query(query)
    cache_chunk_ID = create_chunk_id(doc)
    doc_embedding_entry = vector_cache.get(cache_chunk_ID)
    if not doc_embedding_entry:
        print(f"Warning: could not verify relevance of document {doc.metadata['file_path']}_{doc.metadata.get('page')} (hash {cache_chunk_ID})")
        doc.metadata['score'] = 0
        return True
    doc_embedding = doc_embedding_entry['embedding']
    chunk_score = compute_score(doc_embedding, query_embedding)
    if debug:
        print(f"Score of {doc.metadata['file_path']} {doc.metadata.get('page')} : {chunk_score}")
    doc.metadata['score'] = chunk_score
    return chunk_score >= min_relevance


from sklearn.metrics.pairwise import cosine_similarity

def compute_score(doc_embedding, query_embedding):
    return float(cosine_similarity([query_embedding], [doc_embedding])[0][0])


def removeNonPrintableCharacters(text): #Not used at the moment
    return ''.join(c for c in text if c.isprintable() or c in '\n\r')

def is_hallucinated_by_words(answer: str, context: str, threshold: float = 0.5) -> bool:
    answer_lower = answer.lower()
    context_lower = context.lower()
    wordsInAnswerWOStopWords = [word for word in answer_lower.split() if word not in stop_words]
    matched_words = [word for word in wordsInAnswerWOStopWords if word in context_lower]
    ratio = len(matched_words) / max(len(wordsInAnswerWOStopWords), 1)
    output_string = ""
    for word in answer_lower.split():
        if word in matched_words:
            output_string += word
        else:
            output_string += '_' * len(word)
        index = len(output_string)
        while index < len(answer) and not answer[index].isalpha() :
            output_string += answer[len(output_string)] #Adds whatever character(s) is at that position of the answer, e.g. space or line break
            index += 1
    print(f"\nMatched (non-hallucinated) words:\n{output_string}. (Ratio: {ratio})")
    return ratio < threshold

import re
def extract_ngrams(text: str, n: int=2):
    words = re.findall(r'\w+', text.lower())
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(stopwords.words('german'))


def is_hallucinated_by_phrases(answer: str, context: str, threshold: float = 0.5, ngramsize: int = 2, removeStopwords: bool = True) -> bool:
    answer_lower = answer.lower()
    context_lower = context.lower()
    ngsize_display = ngramsize if ngramsize is not None else 2
    ngrams = extract_ngrams(answer_lower, ngramsize)
    # print(f"ngrams:[{':'.join(ngrams)}]")
    ngrams_without_stopwords = []
    for ng in ngrams:
        if all(word.lower() not in stop_words for word in ng.split()) and ngramContainsNoSymbols(ng):
            ngrams_without_stopwords.append(ng)
    ngrams_to_use = []
    if removeStopwords:
        ngrams_to_use = ngrams_without_stopwords
    else:
        ngrams_to_use = ngrams

    # print(f"ngrams without stop words:[{':'.join(ngrams_to_use)}]")
    matched = [ng for ng in ngrams_to_use if ng in context_lower]
    ratio = len(matched) / max(len(ngrams_to_use), 1)

    print(f"\nMatched (non-hallucinated) {ngsize_display}-grams:\n{', '.join(matched)} (Ratio: {ratio})")
    return ratio < threshold


def ngramContainsNoSymbols(ng):
    if not ng.strip():
        return True
    if not ng[0].isalpha():
        return False
    words = ng.split()
    if len(words) < 2:
        return True
    return words[1].isalpha()


debug = True

# Part of the chain that was done in Embedder.py
# parent document --> chunks --> vectorized into vectorstore (the index.faiss created in the other file is the vectorstore)

# myLLM = TinyOllamaLLMInterface()
myLLM = OllamaLLMInterface()
# myLLM = AzureOpenaiLLMInterface()

if not myLLM.isAvailable():
    print("Sorry, the LLM is not available!")
    exit(1)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)



vectorstore = cast(FAISS, FAISS.load_local( #cast helps the autocomplete to work
    INDEX_NAME,
    embedding_model,
    allow_dangerous_deserialization=True #do not use if you don't trust the source (e.g. you didn't generate the index yourself)
))


prompt_template = """
Do not add meta-comments like 'Here's the answer'.
Answer the question, using only the context below. Do not add any of your own knowledge.
Avoid vague or general statements — draw from concrete details in the context. If the context does not contain an answer, say so clearly.
"""


retriever = vectorstore.as_retriever()      #gets the context
# retriever.search_kwargs["k"] = 1
retriever.search_type = "similarity" #mmr



# This is for checking if the found documents are relevant
import json
import os
vector_cache_name = INDEX_NAME + "/vector_cache_for_relevance_using_" + EMBEDDING_MODEL.replace("/", "_") + ".jsonl"
vector_cache = {}
if os.path.exists(vector_cache_name):
    with open(vector_cache_name, "r") as f:
        for line in f:
            entry = json.loads(line)
            vector_cache[entry["chunk_id"]] = entry
else:
    print("Warning: there is currently no vector cache, so I can't filter out less relevant documents. You may create one using VectorCacheCreator.py")
    print(f"The path I checked: {vector_cache_name}")



# Start the chat

while True:
    user_query = input("Ask a question, or type exit (or Search: term to search):\n")
    if(user_query.lower() in ["exit", "quit"]):
        print("Bye!")
        break
    if(user_query.lower().startswith("search:")):
        search_term = user_query[len("search:"):].strip()
        print("Searching for: [" + search_term + "]")
        pages = find_search_term(search_term, vectorstore)  #Will only find k results, depending on how much you set in the function
        print(f"Pages mentioning '{search_term}':\n",
              "\n".join(str(p) for p in pages))
    else:

        retrieved_docs = retriever.get_relevant_documents(user_query)

        if not retrieved_docs or all(len(doc.page_content.strip()) < 20 for doc in retrieved_docs):
            print("❌ No relevant pages found.")
            continue

        if vector_cache:    #Filtering out documents that aren't relevant enough
            filtered_docs = [doc for doc in retrieved_docs if meets_relevance_criteria(doc, user_query, vector_cache, 0.6)]
        else:
            filtered_docs = retrieved_docs

        if not filtered_docs:
            print("None of the retrieved pages were relevant enough to answer your query")
            continue

        filtered_docs.sort(key=lambda x: x.metadata['score'], reverse=True)

        context_text = "\n\n".join(doc.page_content for doc in filtered_docs)

        if not context_text.strip():
            print("⚠️ No usable context found. Skipping model invocation.")
            continue

        result = None
        try:
            myLLM.setPrompt(prompt_template)
            result = myLLM.respond(query=user_query, context_text=context_text)
        except Exception as e:
            print(f"Got an exception: {e}")
        if not result:
            print("Got no result")
        else:
            print("🔍 Answer:")
            print(result)
            if debug and is_hallucinated_by_words(result, context_text):
                print("Warning! High chance there are hallucinations in this text (based on words)!\n")
            if debug and is_hallucinated_by_phrases(result, context_text, 0.2):
                print("Warning! High chance there are hallucinations in this text (based on 2-word phrases)!\n")
            if debug and is_hallucinated_by_phrases(result, context_text, 0.0, 4, False):
                print("Warning! High chance there are hallucinations in this text (based on 4-word phrases)!")


            print("\n📚 Source pages:")
            for doc in filtered_docs:
                clean_path = doc.metadata['file_path'].replace('\\\\', '\\')
                print(f"Page {doc.metadata['page']} of {clean_path}")
                print("\n\t" + doc.page_content.replace("\n", "\n\t") + "\n")
        print("\n\n")


# Example question: what is a cloud
# C:\Data Science\MAS Data Science\2020FS Big Data\03 HW Architektur Cloud\02_CAS_Big_Data_Infrastructure_Design_FHBE_2020.pdf_4 :