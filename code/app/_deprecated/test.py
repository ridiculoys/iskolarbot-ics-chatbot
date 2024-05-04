# todo: CREATE A SRC FOLDER FOR EVERYTHING AND THEN APP.PY FOR THE MAIN FUNCTION

import chainlit as cl
from templates.search_prompts import SearchPrompts
from helpers.setup import setup_vectorstore, setup_search_content_chain, setup_search_papers_chain, setup_tools, chat_completion_request
from helpers.functions import execute_function_call
import os
import random

#function for the initial messages of the chatbot
async def init_messages():
  actions = [
    cl.Action(name="Artificial Intelligence", label="Artificial Intelligence", value="ics-chatbot-ai", description="Ask questions related to AI"),
    cl.Action(name="Data Structures and Algorithms", label="Data Structures and Algorithms", value="ics-chatbot-algorithms", description="Ask questions related to CS algorithms"),
    cl.Action(name="Cryptography and Security", label="Cryptography and Security", value="ics-chatbot-security", description="Ask questions related to cryptography and security"),
    cl.Action(name="Computer Vision and Pattern Recognition", label="Computer Vision and Pattern Recognition (Under construction)", value="ics-chatbot-computer-vision", description="Ask questions related to computer vision"),
    cl.Action(name="General Questions", label="General Questions", value="ics-chatbot-general", description="Ask general questions about anything about the research papers. This might take longer to answer."),
  ]

  #todo: MultiRetrievalQAChain for the general questions


  WELCOME_MSGS = ["Greetings from the Official ICS Research Chatbot. I am IskolarBot! What fascinating topic are you keen to delve into today?", "Hello there! Welcome aboard the Official ICS Research Chatbot. I'm IskolarBot. Ready to embark on an intellectual journey? What piques your interest today?", "Welcome to the realm of knowledge! I'm IskolarBot, your guide through the wonders of ICS Research Papers. What topic intrigues you today?", "Step into the world of discovery with me, IskolarBot, your friendly neighborhood chatbot for ICS Research Papers. What subject shall we uncover together today?", "Hey, hey! IskolarBot here, your trusty companion for navigating the depths of ICS Research Papers. Which topic should we explore?", "Greetings, curious mind! You've arrived at the Official ICS Research Chatbot, IskolarBot's domain. What topic ignites your curiosity?", "Welcome to the ICS Research Chatbot! I'm IskolarBot, your virtual guide to a plethora of research topics. Where shall we start our exploration today?", "Salutations! IskolarBot at your service, ready to assist you in your quest for knowledge. What subject shall we uncover together today?", "Hello and welcome! I'm IskolarBot, your companion in the world of ICS Research Papers. What aspect of this fascinating field interests you today?", "Greetings, knowledge seeker! You've reached IskolarBot, the official chatbot of ICS Research Papers. What topic would you like to dive into during our conversation today?"]
  content = random.choice(WELCOME_MSGS)

  # Get user input
  res = await cl.AskActionMessage(
    content=content,
    timeout=3600,
    author="IskolarBot Welcome",
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

  # cl.user_session.set("openai_api_key", os.environ["OPENAI_API_KEY"])
  index_name, topic = await init_messages()
  cl.user_session.set("topic", topic)
  cl.user_session.set("index_name", index_name)

  vectorstore = setup_vectorstore(index_name)
  cl.user_session.set("vectorstore", vectorstore)

  chat_history = []
  chat_history.append({"role": "system", "content": f"The content needs to be related to Computer Science as you are a research assistant chatbot. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous. Let the user direct you to the correct answer."})
  cl.user_session.set("chat_history", chat_history)
  
  tools = setup_tools()
  cl.user_session.set("tools", tools)

  await cl.Message(content=f"Completed setting up data. You may now ask your questions related to **{topic}**!", author="IskolarBot").send()

  query_chain = setup_search_content_chain(vectorstore)
  cl.user_session.set("query_chain", query_chain)

from helpers.functions import summarize_paper
@cl.step
async def summarizer(vectorstore, args, user_query, topic, index_name):
    response = await summarize_paper(vectorstore=vectorstore, args=args, user_query=user_query, topic=topic, index_name=index_name)
    print("Response", response)
    
    # Simulate a running task
    # await cl.sleep(2)

    return response

@cl.on_message
async def on_message(msg: cl.Message):
  # user_choice = cl.user_session.get("user_choice")
  vectorstore = cl.user_session.get("vectorstore")
  topic = cl.user_session.get("topic")
  index_name = cl.user_session.get("index_name")
  chat_history = cl.user_session.get("chat_history")

  # Append user message to chat history
  chat_history.append({"role": "user", "content": msg.content})
  print("chat history", chat_history)

  # args = {'paper_title': 'Study and Analysis of Chat GPT and its Impact on Different Fields of Study', 'summary_length': 500, 'focus_on': 'LIMITATIONS AND FEATURES OF CHATGPT'}
  args = {'paper_title': 'Study and Analysis of Chat GPT and its Impact on Different Fields of Study'}

  response = await summarizer(vectorstore=vectorstore, args=args, user_query=msg.content, topic=topic, index_name=index_name)
   
  await cl.Message(content=response, author="IskolarBot").send()
