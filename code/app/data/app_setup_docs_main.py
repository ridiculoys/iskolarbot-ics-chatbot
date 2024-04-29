"""
SETTING UP DOCUMENTS

This script sets up the documents for the AI chatbot. It uses the PyPDFDirectoryLoader library to load PDF files from a directory and then splits the text into chunks. The text chunks are then embedded using the OpenAI API and stored in a Pinecone index. The script uses the Langchain library to handle the document loading, text splitting, and embedding processes.

NOTE: COPY THIS FILE INTO THE ROOT DIRECTORY OF THE DATASET FOLDER
This is because when upserting the document into the Pinecone index, the source of the document is used as the metadata. The source is the file path, which would include the entire directory of the dataset if this file is outside the folder. If the file path is not correct, the metadata will not be accurate in returning the references.
"""

import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
pinecone_api_key=os.environ["PINECONE_API_KEY"]
use_serverless=os.environ["USE_SERVERLESS"]

# todo: edit data path if using outside of the data folder
""" 
OPTIONS:
  ai
  cryptography_security
  datastructures_algorithms
  os
  hci
  general
"""
path_options=["ai", "cryptography_security","datastructures_algorithms", "os", "hci", "general"]
data_path = path_options[0]

# if using within the data folder
data_path = "."

# todo: edit index name
""" 
OPTIONS:
  ics-chatbot-ai
  ics-chatbot-security
  ics-chatbot-algorithms
  ics-chatbot-os
  ics-chatbot-hci
  ics-chatbot-general
"""
index_options = ["ics-chatbot-ai", "ics-chatbot-security", "ics-chatbot-algorithms", "ics-chatbot-os", "ics-chatbot-hci", "ics-chatbot-general"]
index_name=index_options[0]

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.helpers.references import get_references

def setup_documents(data_path, index_name):
  print(f"Loading directory for {index_name}...")
  pdf_loader = PyPDFDirectoryLoader(path=data_path, glob="**/*.pdf", recursive=True)
  documents = pdf_loader.load()

  references = {}
  for document in documents:
    source = document.metadata['source']
    if source in references:
      document.metadata['file_name'] = source
      document.metadata['reference'] = references[source]
      continue
    references[source] = get_references(index_name, source)
    print("Reference:", references[source])

  print('document', documents[0])
  
  print("Directory loaded...")
  print("Documents length:", len(documents))

  print("Splitting text...")
  # Initialize the RecursiveCharacterTextSplitter for splitting text
  # predefined length -- how many chars do we want per chunk
  # overlap - character 0 - 1000, first document. Then, there's an overlap of +-150 characters between doc 1 and doc 2
  text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=50)
  chunked_documents = text_splitter.split_documents(documents)
  print("Chunk", chunked_documents[0])

  print('Length of chunks:', len(chunked_documents))
  
  print("Getting embeddings and vectorstore...")
  embeddings = OpenAIEmbeddings()

  pinecone = PineconeVectorStore.from_documents(
    documents=chunked_documents, embedding=embeddings, index_name=index_name
  )

  return pinecone


print("Pinecone vectorstore start.")
pinecone_vectorstore = setup_documents(data_path, index_name)
print("pinecone", pinecone_vectorstore)
print("Pinecone vectorstore setup complete.")