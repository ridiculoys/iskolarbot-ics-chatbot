# todo: CREATE A SRC FOLDER FOR EVERYTHING AND THEN APP.PY FOR THE MAIN FUNCTION

import chainlit as cl
from templates.search_prompts import SearchPrompts
from helpers.setup import setup_vectorstore, setup_search_content_chain, setup_search_papers_chain
import os

#function for the initial messages of the chatbot
async def init_messages():
  actions = [
    cl.Action(name="Query Research Papers", label="Query Research Papers", value="search_content", description="Ask about the contents of ICS research papers"),
    cl.Action(name="Search for Related Literature", label="Search for Related Literature", value="search_papers", description="Get a list of related research papers"),
    # cl.Action(name="Upload Paper", label="Upload Paper", value="upload_file", description="Upload your own PDF and ask about its contents")
  ]

  content="Welcome to the Official ICS Research Chatbot. You can call me IskolarBot. How can I help you today?"

  # Get user input
  res = await cl.AskActionMessage(
    content=content,
    timeout=3600,
    author="IskolarBot Welcome",
    actions=actions,
  ).send()

  return res.get("value")

async def init_search_content():
  actions = [
    cl.Action(name="Artificial Intelligence", label="Artificial Intelligence", value="ics-chatbot-ai", description="Ask questions related to AI"),
    cl.Action(name="Data Structures and Algorithms", label="Data Structures and Algorithms", value="ics-chatbot-algorithms", description="Ask questions related to CS algorithms"),
    cl.Action(name="Cryptography and Security", label="Cryptography and Security", value="ics-chatbot-security", description="Ask questions related to cryptography and security"),
    cl.Action(name="Computer Vision and Pattern Recognition", label="Computer Vision and Pattern Recognition (Under construction)", value="ics-chatbot-computer-vision", description="Ask questions related to computer vision"),
    cl.Action(name="General Questions", label="General Questions", value="ics-chatbot-general", description="Ask general questions about anything about the research papers. This might take longer to answer."),
  ]

  content="You have chosen to search for the contents of the research papers. What topic would you like to search for?"

  # Get user input
  res = await cl.AskActionMessage(
    content=content,
    timeout=3600,
    author="IskolarBot Action",
    actions=actions,
  ).send()

  return res.get("value"), res.get("label")

@cl.on_chat_start
#todo have a reset chat button, that is the only way to reset the chat
async def on_chat_start():
  # Initialize avatar and name for the chatbot
  await cl.Avatar(
    name="IskolarBot",
    url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4"
  ).send()
  await cl.Avatar(
    name="IskolarBot Welcome",
    url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4"
  ).send()
  await cl.Avatar(
    name="IskolarBot Action",
    url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4"
  ).send()

  cl.user_session.set("openai_api_key", os.environ["OPENAI_API_KEY"])
  user_choice = await init_messages()
  cl.user_session.set("user_choice", user_choice)

  # Handler for user choice -- for future features
  while user_choice != "search_content" and user_choice != "search_papers" : # and user_choice != "upload_file"
    content = "The option you have chosen is still under construction. Please choose another option."
    await cl.Message(content=content, author="IskolarBot").send()
    user_choice = await init_messages()

  index_name, topic = await init_search_content()
  cl.user_session.set("topic", topic)
  cl.user_session.set("index_name", index_name)

  vectorstore = setup_vectorstore(index_name) #change to content
  cl.user_session.set("vectorstore", vectorstore)

  if user_choice == "search_content":
    search_content_chain = setup_search_content_chain(vectorstore)
    cl.user_session.set("chain", search_content_chain)

    await cl.Message(content=f"Completed setting up data. You have chosen to query answers related to **{topic}**. You may now ask your question.", author="IskolarBot").send()

  elif user_choice == "search_papers":
    await cl.Message(content=f"Completed setting up the data. You chose to search for research papers related to **{topic}**. You may now input the topic you want to search for.", author="IskolarBot").send()

  elif user_choice == "upload_file":
    await cl.Message(content="You chose to upload a file.", author="IskolarBot").send()

@cl.on_message
async def on_message(msg: cl.Message):
  user_choice = cl.user_session.get("user_choice")
  vectorstore = cl.user_session.get("vectorstore")
  topic = cl.user_session.get("topic")
  index_name = cl.user_session.get("index_name")

  if user_choice == "search_content":
    chain = cl.user_session.get("chain")
    query = msg.content

    documents = vectorstore.similarity_search(query, k=5)

    context = ""
    for idx, doc in enumerate(documents):
      content = doc.page_content
      filename = doc.metadata['file_name']

      context += f"""
        Document {idx+1}
        Content: {content}
        Filename: {filename}
        ======
      """

    # inputs = {"topic":topic, "question":query, "summaries": vectorstore.as_retriever()}
    inputs = {"topic":topic, "question":query, "summaries": context}
    chain_response = chain.invoke(inputs)
    response = chain_response["answer"]

  elif user_choice == "search_papers":
    response = setup_search_papers_chain(vectorstore=vectorstore, query=msg.content, topic=index_name)

  
  
  await cl.Message(content=response, author="IskolarBot").send()
