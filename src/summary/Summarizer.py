
import sys
import os

from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate

from LLMInterfaces.AzureOpenaiLLMInterface import AzureOpenaiLLMInterface
from LLMInterfaces.OllamaLLMInterface import OllamaLLMInterface
from LLMInterfaces.OpenRouterLLMInterface import OpenRouterLLMInterface

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) #For importing things 1 folder higher
sys.path.insert(0, parent_dir)

from Embedder import getDocsFromIndexByFilter

import gc

llmInterface = OpenRouterLLMInterface()
print(f"LLM used: {llmInterface.getName()}")


def listSources(filtered_documents):
    for doc in filtered_documents:
        set_of_sources.add(doc.metadata.get("source", "unknown"))
    return sorted(set_of_sources)


def removeLLMsPreamble(text):
    lines = text.splitlines()  # Remove empty lines and strip whitespace
    if lines[0] and "summary" in lines[0] or "SUMMARY" in lines[0]:
        meaningful_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(meaningful_lines[1:]) # Skip the first line
    return text


def summarizePage(text: str):
    global result
    template = """
    You are summarizing an academic text on Data Science to create a cheat sheet for future reference. Keep all the relevant key facts.
    Output only a summary of the following text in 1-5 sentences, without any introduction, greetings, preamble, commentary or explanation (i.e. do NOT write "here's the summary you requested" or similar).
    If the page is showing a fictional example (marked with the title "Beispiel" or similar) to illustrate the math, mention that it is an example:\\n\\n{context}
    If no text was given to summarize, return an empty string.
    Output:
    [SUMMARY]
    <your summary here>
    """


    llmInterface.setPrompt(template)
    summaryFromLLM = llmInterface.respond(query="", context_text=text)
    summary = removeLLMsPreamble(summaryFromLLM) #If stupid LLM added "Sure, here's a summary"

    return summary

def summarizeDocument(text: str):
    global result
    template = """
    Output a summary of the text. Mention all key concepts the text addresses (and their names in the language they are originally mentioned in). 
    Structure the summary into sections only if the original text clearly separates topics (e.g., via headings or thematic shifts). Do not invent chapters or content not explicitly present.
    If there is an index in the beginning, include it in the summary.
    Ignore mentions of moodle, Berner Fachhochschule or Bern university, illustrative examples, citations, or rhetorical questions unless they are central to the argument.\\n\\n{context}
    Output:
    [SUMMARY]
    <your summary here>
    """
    llmInterface.setPrompt(template)
    summaryFromLLM = llmInterface.respond(query="", context_text=text)
    summary = f"""
    =======================
    Prompt used: 
    {template}
    =======================

    """\
              + removeLLMsPreamble(summaryFromLLM) #If stupid LLM added "Sure, here's a summary"

    return summary

def create_chunk_id(doc):
    return f"{doc.metadata['file_path']}_{doc.metadata.get('page')}"


def combineTextByDocument(filtered_documents):
    docContents = {}
    for doc in filtered_documents:  # per page
        existingDoc = docContents.get(doc.metadata['file_path'])
        if not existingDoc:
            existingDoc = {}
        existingDoc[doc.metadata.get('page')] = doc.page_content
        docContents[doc.metadata['file_path']] = existingDoc
    for path, text in docContents.items():
        sorted_pages = dict(sorted(text.items()))
        combined_text = "\n".join(sorted_pages.values())
        docContents[path] = combined_text
    return docContents

def summarizeEachPageOfEachDoc(filtered_documents):
    summaries = {}
    for doc in filtered_documents:  # per page
        existingDoc = summaries.get(doc.metadata['file_path'])
        if not existingDoc:
            existingDoc = {}
        summ = summarizePage(doc.page_content)
        existingDoc[doc.metadata.get('page')] = summ
        summaries[doc.metadata['file_path']] = existingDoc
    return summaries

from typing import Dict

def summarizeDocumentViaPageSummaries(summaries: Dict[str, Dict[int, str]]) -> Dict[str, str]:
    docSummaries = {}
    for path, pageSums in sorted(summaries.items()):
        textOfSummaries = ""
        for page, pageSum in pageSums.items():
            textOfSummaries += pageSum + "\n"
            # print(f"{page} → {pageSum}")
        print(f"Summary of {path}\n{textOfSummaries}")
        docSum = summarizeDocument(textOfSummaries)
        docSummaries[path] = docSum
        print(f"\nThe summary is: {docSum}")
    return docSummaries


def getSummaryOutputFolder(summary_output_folder):
    # Dirname prevents a folder named bla.pdf from being created. Uses only folders.
    _, relative_path = os.path.splitdrive(os.path.dirname(summary_output_folder)) # Splitdrive removes "C:\ etc.
    relative_path = relative_path.replace("\\", "/").lstrip("/")
    summary_output_folder = os.path.normpath(os.path.join("outputs", relative_path))
    return summary_output_folder

import re
def summarizeAndSaveEachFile_fromPageSummaries(summaries_of_Pages):
    for path, pageSums in summaries_of_Pages.items():
        textOfSummaries = ""
        for page, pageSum in pageSums.items():
            textOfSummaries += pageSum + "\n"
            # print(f"{page} → {pageSum}")
        print(f"Summary of {path}\n{textOfSummaries}")
        docSum = summarizeDocument(textOfSummaries)
        folder_to_save = getSummaryOutputFolder(path)
        os.makedirs(folder_to_save, exist_ok=True)

        name_of_original_file = os.path.basename(path)
        suffix = "_byPagesThenOverall.txt"
        max_path_name = 195 #found this by experimenting. Don't know why it's this value
        max_file_name = max_path_name - len(folder_to_save) - len(llmInterface.getName()) - len(suffix) - len(" ") - 1
        if len(name_of_original_file) > max_file_name:
            name_of_original_file = shortenFileName(name_of_original_file, max_file_name)

        filename = name_of_original_file \
                   + " " + re.sub(r'[<>:"/\\|?*]', '_', llmInterface.getName()) \
                   + suffix
        full_path = os.path.join(folder_to_save, filename)
        with open(full_path, "w", encoding="utf-8") as file:
            file.write(docSum)
        print(f"\nSaved: {full_path}")


def shortenFileName(name_of_original_file, max_file_name):
    charsToRemove = len(name_of_original_file)-max_file_name
    charsWeCheckedForRemoval = 0
    cutoffPoint = 1
    for i in range(len(name_of_original_file)-1, -1, -1):
        if not name_of_original_file[i].isdigit():      #leave the digits in, since they're likely a numbering that helps context
            charsWeCheckedForRemoval = charsWeCheckedForRemoval + 1
        if charsWeCheckedForRemoval >= charsToRemove:
            cutoffPoint = i
            break

    if cutoffPoint == 1: #too many digit-characters to spare them
        cutoffPoint = -charsToRemove

    return name_of_original_file[0:cutoffPoint]


def summarizeAndSaveEachFile_fromFullContent(fullTexts):
    for path, content in fullTexts.items():
            docSum = summarizeDocument(content)
            folder_to_save = getSummaryOutputFolder(path)
            os.makedirs(folder_to_save, exist_ok=True)
            name_of_original_file = os.path.basename(path)
            suffix = "_fromWholeContent.txt"
            max_path_name = 195 #found this by experimenting. Don't know why it's this value
            max_file_name = max_path_name - len(folder_to_save) - len(llmInterface.getName()) - len(suffix) - len(" ") - 1
            if len(name_of_original_file) > max_file_name:
                name_of_original_file = shortenFileName(name_of_original_file, max_file_name)

            filename = name_of_original_file \
                       + " " + re.sub(r'[<>:"/\\|?*]', '_', llmInterface.getName()) \
                       + suffix
            full_path = os.path.join(folder_to_save, filename)

            print(f"Path length: {len(full_path)}")


            with open(full_path, "w", encoding="utf-8") as file:
                file.write(docSum)
            print(f"\nSaved: {full_path}")


def summarizeFirstByPageThenByDocumentAndSave(filtered_documents):
    summaries_of_Pages = summarizeEachPageOfEachDoc(filtered_documents)
    summarizeAndSaveEachFile_fromPageSummaries(summaries_of_Pages)
    del summaries_of_Pages
    gc.collect()


def summarizeDirectlyByDocumentAndSave(filtered_documents):
    fullTexts = combineTextByDocument(filtered_documents)
    summarizeAndSaveEachFile_fromFullContent(fullTexts)


if __name__ == "__main__":

    if not llmInterface.isAvailable():
        print("llmInterface called " + llmInterface.getName() + " is not available")
        exit(1)

    # folder_to_summarize = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\Wahltag Kausalanalyse\kausalanalyse.pdf'
    # folder_to_summarize = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\ExpVis\Handouts\CAS Datenanalyse_Datenvisualisierung_0 Einführung'
    folder_to_summarize = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\ExpVis\Handouts\CAS Datenanalyse_Datenvisualisierung_1 Grafische Datenexploration'
    # folder_to_summarize = r'C:\Data Science\MAS Data Science\2019FS Datenanalyse\ExpVis\Handouts\CAS Datenanalyse_Datenvisualisierung_2 Grafiken als Mittel der Kommunikation.pdf'


    summary_output_folder = getSummaryOutputFolder(folder_to_summarize)
    os.makedirs(summary_output_folder, exist_ok=True)

    filtered_documents = getDocsFromIndexByFilter(folder_to_summarize, 0, 5)
    filtered_documents = sorted(filtered_documents, key=lambda x: x.metadata.get("source"))
    print(f"got {len(filtered_documents)} documents")

    #For debugging
    set_of_sources = set()
    listOfSources = listSources(filtered_documents)
    print("List of sources that appear")
    print("\n".join(listOfSources))

    # summarizeFirstByPageThenByDocumentAndSave(filtered_documents)
    
    summarizeDirectlyByDocumentAndSave(filtered_documents)
