
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# "gpt-4-1106-preview"
GPT_MODEL = "gpt-3.5-turbo-0125"
EMBEDDING_MODEL = "text-embedding-ada-002"

def setup_vectorstore(index_name):
  """
  20 serverless indexes max == 20 topics. Each serverless index is limited to 10,000 namespaces. Each serverless namespace is limited to 1B vectors.
  """

  """
  text-embedding-ada-002 = standard
  text-embedding-3-small = optimized for latency and storage, efficiency is a priority. dimension of 1536 and can handle up to 8191 tokens
  text-embedding-3-large: higher accuracy. dimension of 3072, which is larger than the 1536-dimensional version of text-embedding-3-small.
  """
  embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
  pinecone_vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

  return pinecone_vectorstore

from app.templates.search_prompts import SearchPrompts
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory

#todo: try not to use the retrieval qa with soruces cos i don't think we need the sources if we can get the metadata from the vector similarity search
def setup_search_content_chain(pinecone_vectorstore, template=SearchPrompts.answer_query_prompt()):
  llm = ChatOpenAI(model=GPT_MODEL, temperature=0)

  # template = SearchPrompts.answer_query_prompt() #todo update prompt -- sometimes it includes the "Answer" as a title
  
  #This one creates a summary of the whole interaction
  # memory = ConversationSummaryBufferMemory(llm=model, input_key='question', output_key='answer')

  # somehow worse in the hallucination
  # might need to include a prompt that is more specific to the memory
  # memory = ConversationBufferMemory(memory_key="chat_history", input_key='question', output_key='answer', return_messages=True)

  chain = RetrievalQAWithSourcesChain.from_chain_type(
      llm=llm,
      chain_type="stuff",
      retriever=pinecone_vectorstore.as_retriever(),
      chain_type_kwargs={
          "prompt": template,
          "verbose": True,
          # "memory": ConversationBufferMemory(
          #   memory_key='history',
          #   input_key='question'),
      },
      # memory=memory, #makes it hallucinate a bit
      verbose=True,
  )

  return chain


def setup_search_papers_chain(vectorstore, query):
  results = vectorstore.similarity_search_with_score(query, k=10)
  
  filenames = list(set([result[0].metadata['file_name'] for result in results]))
  added = []

  references = []
  for result in results:
    if result[0].metadata['file_name'] in filenames and result[0].metadata['file_name'] not in added:
      references.append(result[0].metadata)
      added.append(result[0].metadata['file_name'])

  response = ""
  for idx, reference in enumerate(references):
    response += f"[{idx+1}] {reference['reference']}\n"

  return response


from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
async def setup_summary_chain(index_name, filename):
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

  index = index_options.index(index_name)
  data_path = f"app/data/{path_options[index]}/{filename}"
  # data_path = f"{filename}"
  # print("data_path", data_path)
  llm = ChatOpenAI(model=GPT_MODEL, temperature=0)
  
  loader = PyPDFLoader(data_path)
  docs = loader.load()

  # text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=50)
  # chunked_documents = text_splitter.split_documents(docs)
  
  # print(f"Loading {len(docs)} documents")
  
  chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
  # chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, verbose=True)

  # arun is deprecated, use ainvoke
  # summary = await chain.arun(docs)   # better response = docs
  # summary = await chain.arun(chunked_documents)   
  # return summary

  summary = await chain.ainvoke(docs)  
  # summary = await chain.ainvoke(chunked_documents)   
  return summary['output_text']

# OpenAI Function Calling
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
client = OpenAI()

# Utilities
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
# Utilities

# OpenAI Function Calling
from app.templates.formatter_prompts import FormatterPrompts
def setup_tools():
  # [ICEBOX] more tools - tech stack recommender, 
  #todo: refine descriptions -- summarizer needs to parse author properly

  #todo: add semantic keywords instead of topic

  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_answer",
        "description": "Use this function to provide direct answers to the user's general queries or inquiries, such as definitions, clarifications, context of something, explanations, follow-ups, examples, and more. These include but are not limited to: `What is..`, `What are the types of..`, `How does... work?`, `What are the latest findings on..`, `Can you provide examples of..`, `What methodologies are used to study..`, `Why is..`, and other variations of these. Use this function if the query is generally asking for something that will be answered by a study or research paper. It is designed to provide detailed answers to support the user's understanding of the subject matter.",
        "parameters": {
          "type": "object",
          "properties": {
            "question_type": {
              "type": "string",
              "description": "The nature of the user's inquiry if it is asking for clarification, context, definition, explanation, follow-up, comparison, results, or any other type of inquiry."
            },
            "question_subject": {
              "type": "string",
              "description": "The specific topic or subject within the research paper or journal that the user is inquiring about."
            },
            "semantic_keywords": {
              "type": "string",
              "description": "The semantic keywords associated with the user query. Do not make assumptions from the query. Return as a comma-separated list."
            }
          },
          "required": ["question_type", "question_subject", "semantic_keywords"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "summarize_paper",
        "description": "Use this function ONLY if the user is asking for a summary. This function will be used to extract the key points, main ideas, and findings from the paper, providing a clear overview that can be used for quick understanding.",
        "parameters": {
          "type": "object",
          "properties": {
            "paper_title": {
              "type": "string",
              "description": "The title of the journal or research paper to be summarized."
            },
            "paper_authors": {
              "type": "string",
              "description": "The author(s) of the journal or research paper to be summarized."
            },
            "subject": {
              "type": "string",
              "description": "A specific subject, aspect, topic, or section of the paper that the user asks for. It could be about a specific part of the paper such as the introduction, related literature, methodology, results, conclusion, or it could be a topic they want to highlight."
            },
            "semantic_keywords": {
              "type": "string",
              "description": "The semantic keywords associated with the user query. Do not make assumptions from the query. Return as a comma-separated list."
            }
          },
          "required": ["paper_title", "semantic_keywords"],
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_related_literature",
        "description": "Use this function ONLY if the user specifically asks to find or get related literature about a specific topic. Examples include `Give me or find papers about...`, `What studies have been conducted regarding...`, `Are there any research articles available that touch on...`, ```I'm seeking papers related to...` and other variations of this. This function will be used to search for articles, papers, and other scholarly works that are closely related to the provided topic.",
        "parameters": {
          "type": "object",
          "properties": {
            "topic": {
              "type": "string",
              "description": "The specific topic for which related literature is sought. This could be a research question, a concept, a field of study, or any other topic of interest."
            },
            "semantic_keywords": {
              "type": "string",
              "description": "The semantic keywords associated with the user query. Do not make assumptions from the query. Return as a comma-separated list."
            }
          },
          "required": ["topic", "semantic_keywords"],
        }
      }
    }
  ]

  return tools