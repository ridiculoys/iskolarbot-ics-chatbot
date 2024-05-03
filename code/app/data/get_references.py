
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai.chat_models import ChatOpenAI

from langchain.chains.combine_documents import create_stuff_documents_chain

from formatter_prompts import FormatterPrompts
    
def get_references(topic, pdf_name):
  topics = ["ics-chatbot-ai", "ics-chatbot-algorithms", "ics-chatbot-security", "ics-chatbot-computer-vision", "ics-chatbot-general"]
  data_path = ["ai", "datastructures_algorithms", "cryptography_security", "computer_vision", "general"]
  topic_index = topics.index(topic)

  # Load the PDF
  loader = PyPDFLoader(f"data/{data_path[topic_index]}/{pdf_name}.pdf")
  document = loader.load_and_split()

  # Get only the first 3 pages
  page_len = 3 if len(document) > 3 else len(document)
  pages = document[:page_len]

  content = [page.page_content for page in pages]
  content = " ".join(content)

  model = ChatOpenAI(model='gpt-3.5-turbo-0125', temperature=0)

  template = FormatterPrompts.ieee_ref_prompt()
  chain = create_stuff_documents_chain(model, template)
  response = chain.ainvoke({"context": pages})
  
  return response