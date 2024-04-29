
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


from app.helpers.references import get_references
# todo: if people want just related papers, i think only need chain for the formatting, the sim_search can handle the getting of the papers
# new todo: since we have the reference in the metadata, only get the sim search and get metadata -- no need for chain
def setup_search_papers_chain(vectorstore, query, topic):
  # results = vectorstore.similarity_search(query, k=50)
  results = vectorstore.similarity_search_with_score(query, k=10)
  
  for idx,result in enumerate(results):
    print(idx, result)
  metadata = [result[0].metadata for result in results]
  print("metadaa", metadata)

  # todo: for each document answer, get metadata to use for references
  # filenames = list(set([result.metadata['source'].replace("data\\ai\\", "").rstrip('.pdf') for result in results]))
  # print("filenames", filenames)

  filenames = list(set([result[0].metadata['file_name'].rstrip('.pdf') for result in results]))

  # put this in setup docs chain
  # response = f"Sure! Here are some related literature I found for **{query}**.\n\n"
  # for idx, filename in enumerate(filenames):
  #   ref = get_references(topic, filename)
  #   response += f"**[{idx+1}]** {ref}\n"

  # return response
  return "\n".join(filenames)


from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
async def setup_summary_chain(index_name, filename):
  """ 
  OPTIONS:
    ai
    datastructures_algorithms
    cryptography_security
    computer_vision
    general
  """
  path_options=["ai", "datastructures_algorithms", "cryptography_security", "computer_vision", "general"]

  """ 
  OPTIONS:
    ics-chatbot-ai
    ics-chatbot-algorithms
    ics-chatbot-security
    ics-chatbot-computer-vision
    ics-chatbot-general
  """
  index_options = ["ics-chatbot-ai", "ics-chatbot-algorithms", "ics-chatbot-security", "ics-chatbot-computer-vision", "ics-chatbot-general"]

  index = index_options.index(index_name)
  data_path = f"data/{path_options[index]}/{filename}"
  # data_path = f"{filename}"
  print("data_path", data_path)
  llm = ChatOpenAI(model=GPT_MODEL, temperature=0)
  
  loader = PyPDFLoader(data_path)
  docs = loader.load()

  # text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=1000, chunk_overlap=50)
  # chunked_documents = text_splitter.split_documents(docs)
  
  print(f"Loading {len(docs)} documents")
  
  chain = load_summarize_chain(llm, chain_type="map_reduce")

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
def setup_tools():
  # [ICEBOX] more tools - tech stack recommender, 
  #todo: refine descriptions -- summarizer needs to parse author properly
  tools = [
    {
      "type": "function",
      "function": {
        "name": "get_answer",
        "description": "Use this function to provide direct answers to the user's general queries or inquiries, such as definitions, clarifications, context of something, explanations, follow-ups, examples, and more. These include but are not limited to: `What is..`, `What are the types of..`, `How does... work?`, `What are the latest findings on..`, `Can you provide examples of..`, `What methodologies are used to study..`, `Why is..`, and other variations of these. Use this function if the query is generally asking for something that will be answered by a study or research paper. It is designed to provide detailed answers to support the user's understanding of the subject matter.",
        "parameters": {
          "type": "object",
          "properties": {
            "type_of_question": {
              "type": "string",
              "description": "The nature of the user's inquiry if it is asking for clarification, context, definition, explanation, follow-up, comparison, results, or any other type of inquiry."
            },
            "paper_topic": {
              "type": "string",
              "description": "The specific topic or subject within the research paper or journal that the user is inquiring about."
            },
            "query_details": {
              "type": "string",
              "description": "Additional details or specific aspects of the topic the user wants to know more about."
            }
          },
          "required": ["type_of_question", "paper_topic"]
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
            "summary_length": {
              "type": "integer",
              "description": "The desired length of the summary in words. Do not assume length if not specified."
            },
            "focus_on": {
              "type": "string",
              "description": "A specific aspect or section of the paper to focus on for the summary. Options include 'introduction', 'related literature', 'methodology', 'results', 'conclusion', or any other section of the paper."
            }
          },
          "required": ["paper_title"],
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
            "limit": {
              "type": "integer",
              "description": "Optional. The maximum number of related literature items to return. Default is 3 if not specified."
            }
          },
          "required": ["topic"],
        }
      }
    }
  ]

  return tools