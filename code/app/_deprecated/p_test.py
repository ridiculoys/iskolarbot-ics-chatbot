import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
pinecone_api_key=os.environ["PINECONE_API_KEY"]
use_serverless=os.environ["USE_SERVERLESS"]
index_name=os.environ["PINECONE_INDEX_NAME"]


from langchain_openai.chat_models import ChatOpenAI

model = ChatOpenAI(openai_api_key=openai_api_key, model='gpt-3.5-turbo')


from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
pinecone_vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
print("pinecone_" , pinecone_vectorstore)

from templates.search_prompts import SearchPrompts
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
template = SearchPrompts.answer_query_prompt() #todo update 
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=pinecone_vectorstore.as_retriever(),
    chain_type_kwargs={
        "prompt": template,
    },
    # memory=memory, #makes it hallucinate a bit
    verbose=True,
  )
print("chain", chain)

user_input=input("Enter your question: ")
inputs = {"topic":"Artificial Intelligence", "question":user_input, "summaries": pinecone_vectorstore.as_retriever()}
chain_response = chain.invoke(inputs)
response = chain_response["answer"]
print("Response:", response)