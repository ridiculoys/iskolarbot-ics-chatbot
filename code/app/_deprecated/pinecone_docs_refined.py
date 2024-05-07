import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
pinecone_api_key=os.environ["PINECONE_API_KEY"]
use_serverless=os.environ["USE_SERVERLESS"]

#Creating an index
# from pinecone import Pinecone, ServerlessSpec, PodSpec

# # configure client
# pc = Pinecone(api_key="9e98cfae-38b1-472f-bc69-413c1ee158cb")

# if use_serverless:
#   spec = ServerlessSpec(cloud='aws', region='us-west-2')
# else:
#   # if not using a starter index, you should specify a pod_type too
#   spec = PodSpec()
#   # spec = PodSpec(environment=environment)

index_name=os.environ["PINECONE_INDEX_NAME"]
# #deletes if exists 
# if index_name in pc.list_indexes().names():
#   pc.delete_index(index_name)

# import time

# dimension = 1536 #768 or 1536
# pc.create_index(
#   name=index_name,
#   dimension=dimension,
#   metric="cosine",
#   spec=spec
# )

# while not pc.describe_index(index_name).status['ready']:
#   time.sleep(1)

# print("hello")

# index = pc.Index(index_name)
# index.describe_index_stats()
# print("Index stats", index)


# https://www.youtube.com/watch?v=BrsocJb-fAo

data_path = "data/ai"

from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo')

#with parser, only receive the content
from langchain_core.output_parsers import StrOutputParser 
parser=StrOutputParser()

from templates.search_prompts import SearchPrompts
prompt = SearchPrompts.get_related_papers_prompt()
print("Done loading prompt")

#with prompt templates

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# new
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
def load_documents():
  # print("Loading documents from PDFs")
  # documents = []
  # for file in os.listdir(data_path):
  #   if file.endswith('.pdf'):
  #     pdf_path = os.path.join(data_path, file)
  #     loader = PyPDFLoader(pdf_path)
  #     documents.extend(loader.load())

  pdf_loader = PyPDFDirectoryLoader(path=data_path, glob="**/*.pdf", recursive=True)
  print("Done loading directory")
  documents = pdf_loader.load()
  
  print("Docs length", len(documents))
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10) #new
  chunked_documents = text_splitter.split_documents(documents)

  # Remove the bibiography of each PDF ?? Maybe not needed
  print('len of chunks', len(chunked_documents))
  # print("chunks", chunked_documents[1])

  # Initialize the RecursiveCharacterTextSplitter for splitting text
  # predefined length -- how many chars do we want per chunk
  # overlap - character 0 - 1000, first document. Then, there's an overlap of +-150 characters between doc 1 and doc 2
  # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

  print("getting embeddings and vectorstore")
  embeddings = OpenAIEmbeddings()
  pinecone = PineconeVectorStore.from_documents(
    documents=chunked_documents, embedding=embeddings, index_name=index_name
  )

  return pinecone


# pinecone_vectorstore = load_documents()

embeddings = OpenAIEmbeddings()
pinecone_vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
print("done getting vectorstore")

# query = "What is Langchain? When you answer, give the referenced source papers for this."
query = "What is langchain?"
# retrieve 4 most similar documents
sim_search = pinecone_vectorstore.similarity_search(query, k=4)
print("sim_search", sim_search[0])

# todo: for each document answer, get metadata to use for references
# todo: if people want just related papers, i think only need chain for the formatting, the sim_search can handle the getting of the papers
first_doc = sim_search[0]
print("metadata", first_doc.metadata)


# parse metadata to extract the source papers
# Add csv agent here to create the reference



#new
from langchain.chains.question_answering import load_qa_chain
# from langchain.memory import ConversationBufferMemory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# from langchain.chains import ConversationalRetrievalChain
# qa = ConversationalRetrievalChain.from_llm(
#     model,
#     pinecone_vectorstore,
#     prompt=prompt,
#     # memory=memory
# )

chain = load_qa_chain(llm=model, chain_type="stuff")

user_input = input("Ask a question: ")

while user_input != "0":
  answer = chain.run(input_documents=sim_search, question=query)
  print("Res:", answer)
  user_input = input("Ask a question: ")