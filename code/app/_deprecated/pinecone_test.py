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

#vanilla
# question = "Who is the president of the Philippines?"
# model.invoke(question)
#vanilla

#with parser, only receive the content
from langchain_core.output_parsers import StrOutputParser 

parser=StrOutputParser()

# chain = model | parser
# question = "Who is the president of the Philippines?"
# model.invoke(question)
#with parser

#with prompt templates
# from langchain_core.output_parsers import StrOutputParser 
# parser=StrOutputParser()

from templates.search_prompts import SearchPrompts
prompt = SearchPrompts.get_related_papers_prompt()
print("Done loading prompt")

# just to see the vars into the prompt
# prompt.format(topic="Artificial Intelligence", question="What are LLMs?")

# # invoking
# chain = prompt | model | parser
# print("here", chain.invoke({
#   "topic": "Artificial Intelligence",
#   "question": "What are LLMs?"
# }))

#with prompt templates


from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the PyPDFDirectoryLoader with the directory containing your PDF files
pdf_loader = PyPDFDirectoryLoader(path=data_path, glob="**/*.pdf", recursive=True)
print("Done loading directory")

# Initialize the RecursiveCharacterTextSplitter for splitting text
# predefined length -- how many chars do we want per chunk
# overlap - character 0 - 1000, first document. Then, there's an overlap of +-150 characters between doc 1 and doc 2
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Load the PDF files
docs = pdf_loader.load()

print("Docs length", len(docs))
# Split each document into chunks
split_docs = text_splitter.split_documents(docs)
# print('here', split_docs)

# Remove the bibiography of each PDF ?? Maybe not needed
print('len of chunks', len(split_docs))
print("chunks", split_docs[1])

from langchain_openai.embeddings import OpenAIEmbeddings

# embeddings = OpenAIEmbeddings()
# embedded_query= embeddings.embed_query("What is an LLM? Give me one example.")

# print(f"Embedding length: {len(embedded_query)}")
# print(f"Embedding: {embedded_query[:10]}")

from langchain_pinecone import PineconeVectorStore

# pinecone = PineconeVectorStore.from_documents(
#   documents=split_docs, embedding=embeddings, index_name=index_name
# )
embeddings = OpenAIEmbeddings()
pinecone = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
print("done getting vectorstore")

sim_search = pinecone.similarity_search("What are LLMs?")[:3]
print("sim_search", sim_search)

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

# prompt.format(topic="Artificial Intelligence", input="What are LLMs?")

setup = RunnableParallel(context=pinecone.as_retriever(), input=RunnablePassthrough())
chain = (
    # {
    #   "context": pinecone.as_retriever(),
    #   "topic": itemgetter("topic"),
    #   "question": itemgetter("input"),
    # }
    setup
    | prompt
    | model
    | StrOutputParser()
)
print("done setting up chain")

print("chain", chain)

# res = chain.invoke("Langchain")
# inputs = {"input":"What are LLMs?", "topic":"Artificial Intelligence"}
# res = chain(inputs)
# res = chain.invoke({
#   "context": pinecone.as_retriever(),
#   "topic": "Artificial Intelligence",
#   "input": "Give me list of papers about LLMs"
#   })
# print("response:", res)


# todo: make sure it is calling from the database, use metadata
# todo: use RetrievalQAWithSourceChain
#https://github.com/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/05-langchain-retrieval-augmentation.ipynb

# CURRENTLY -- Getting data from the references instead of the PDF itself
# use the metadata to get the data from the PDF - use CSV ?? for getting more accurate results?




# ==========================
# for retrieving with qa chain



from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory
template = SearchPrompts.answer_query_prompt()
print("got template")
# memory = ConversationSummaryBufferMemory(llm=model, input_key='question', output_key='answer')

memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=model,
    chain_type="stuff",
    memory=memory,
    retriever=pinecone.as_retriever(),
    chain_type_kwargs={
        "prompt": template,
    },
    verbose=True
)
print("created chain")

user_input = input("Enter your question: ")

while user_input != "0":
    inputs = {"topic":"Artificial Intelligence", "question":user_input, "summaries": pinecone.as_retriever()}
    res = chain.invoke(inputs)
    print("Full Response:", res)
    print("=====")
    print("Answer:", res["answer"])
    print("Sources:", res["sources"])
    user_input = input("Enter your question: ")