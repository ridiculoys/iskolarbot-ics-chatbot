import chainlit as cl
import random
from app.helpers.setup import setup_vectorstore, setup_tools, setup_search_content_chain, chat_completion_request
from app.helpers.functions import execute_function_call
from app.templates.summary_prompts import SummaryPrompts
from app.templates.system_prompts import SystemPrompts

#function for the initial messages of the chatbot
async def init_messages():
  actions = [
    cl.Action(name="Artificial Intelligence", label="Artificial Intelligence", value="ics-chatbot-ai", description="Ask questions related to AI"),

    cl.Action(name="Cryptography and Security", label="Cryptography and Security", value="ics-chatbot-security", description="Ask questions related to cryptography and security"),

    cl.Action(name="Data Structures and Algorithms", label="Data Structures and Algorithms", value="ics-chatbot-algorithms", description="Ask questions related to CS algorithms"),

    cl.Action(name="Operating Systems", label="Operating Systems", value="ics-chatbot-os", description="Ask questions related to operating systems"),

    cl.Action(name="Human-Computer Interaction", label="Human-Computer Interaction", value="ics-chatbot-hci", description="Ask questions related to human-computer interaction (HCI)"),

    cl.Action(name="General Questions", label="General Questions", value="ics-chatbot-general", description="Ask general questions about anything about the research papers. The answers here might take longer and might not be as accurate."),
  ]

  WELCOME_MSGS = [
    "Greetings from the Official ICS Research Chatbot. I am IskolarBot! What fascinating topic are you keen to delve into today? If this is your first time here, be sure to check out the README tab for important information!",
    "Hello there! Welcome aboard the Official ICS Research Chatbot. I'm IskolarBot. Ready to embark on an intellectual journey? What piques your interest today? Don't forget to read the README tab if you're new to this app!",
    "Welcome to the realm of knowledge! I'm IskolarBot, your guide through the wonders of ICS Research Papers. What topic intrigues you today? New to this app? Make sure to read the README tab for essential info!",
    "Step into the world of discovery with me, IskolarBot, your friendly neighborhood chatbot for ICS Research Papers. What subject shall we uncover together today? If you're just starting out, check out the README tab for guidance!",
    "Hey, hey! IskolarBot here, your trusty companion for navigating the depths of ICS Research Papers. Which topic should we explore? If you're new here, don't forget to read the README tab!",
    "Greetings, curious mind! You've arrived at the Official ICS Research Chatbot, IskolarBot's domain. What topic ignites your curiosity? If you're unfamiliar with this app, be sure to check out the README tab!",
    "Welcome to the ICS Research Chatbot! I'm IskolarBot, your virtual guide to a plethora of research topics. Where shall we start our exploration today? New here? Check out the README tab for important information!",
    "Salutations! IskolarBot at your service, ready to assist you in your quest for knowledge. What subject shall we uncover together today? If you're a newcomer, take a look at the README tab for guidance!",
    "Hello and welcome! I'm IskolarBot, your companion in the world of ICS Research Papers. What aspect of this fascinating field interests you today? New to this app? Be sure to read the README tab for essential info!",
    "Greetings, knowledge seeker! You've reached IskolarBot, the official chatbot of ICS Research Papers. What topic would you like to dive into during our conversation today? If you're just starting out with this app, check out the README tab for guidance!"
  ]

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
async def on_chat_start():
  # Initialize avatar and name for the chatbot
  url = "https://cdn.dribbble.com/users/2214460/screenshots/4536093/media/e406d7aa69b9f1f880c94483e0f3d91f.png?resize=400x300&vertical=center"
  await cl.Avatar(name="IskolarBot", url=url).send()
  await cl.Avatar(name="IskolarBot Welcome", url=url).send()
  await cl.Avatar(name="IskolarBot Action", url=url).send()

  # Initial messages to get the topic and the index name for the vectorstore
  index_name, topic = await init_messages()
  cl.user_session.set("topic", topic)
  cl.user_session.set("index_name", index_name)

  vectorstore = setup_vectorstore(index_name)
  cl.user_session.set("vectorstore", vectorstore)

  # Chat history will be used for memory
  chat_history = []
  system_prompt = SystemPrompts.initial_system_prompt()
  chat_history.append({"role": "system", "content": system_prompt })
  cl.user_session.set("chat_history", chat_history)

  #todo: MultiRetrievalQAChain for the general questions
  # if index_name == "ics-chatbot-general":
  #   await cl.Message(content=f"Completed setting up data. You may now ask your general questions!", author="IskolarBot").send()
  #   return
  
  # Setting up the tools for the function call
  tools = setup_tools()
  cl.user_session.set("tools", tools)

  # Setting up the search content chain
  query_chain = setup_search_content_chain(vectorstore)
  cl.user_session.set("query_chain", query_chain)

  # Setting up summary chain
  summary_template = SummaryPrompts.summary_prompt()
  summary_chain = setup_search_content_chain(vectorstore, template=summary_template)
  cl.user_session.set("summary_chain", summary_chain)

  await cl.Message(content=f"Completed setting up data. You may now ask your questions related to **{topic}**!", author="IskolarBot").send()


@cl.on_message
async def on_message(msg: cl.Message):
  # get the necessary data from the user session
  vectorstore = cl.user_session.get("vectorstore")
  topic = cl.user_session.get("topic")
  index_name = cl.user_session.get("index_name")
  chat_history = cl.user_session.get("chat_history")
  tools = cl.user_session.get("tools")

  # Append user message to chat history
  chat_history.append({"role": "user", "content": msg.content})

  # Double check if the user response is answerable by the vectorstore
  response = ""

  chat_response = chat_completion_request(
      messages=chat_history, tools=tools
  )

  assistant_message = chat_response.choices[0].message

  # if tools are used
  if assistant_message.tool_calls:
    # do function call here
    response = await execute_function_call(vectorstore=vectorstore, message=assistant_message, user_query=msg.content, topic=topic, index_name=index_name, chat_history=chat_history)
    print("results", response)

    function_data = {"role": "function", "tool_call_id": assistant_message.tool_calls[0].id, "name": assistant_message.tool_calls[0].function.name, "content": response}
    chat_history.append(function_data)
    print("function data", chat_history)
    
    chat_history.append({"role": assistant_message.role, "content": response})
  else:
    response = assistant_message.content
    chat_history.append({"role":  assistant_message.role, "content": assistant_message.content})

  system_msg = chat_history[0]
  last_four_msgs = chat_history[-4:]
  # Combine the first item with the last 4 items
  chat_history = [system_msg] + last_four_msgs
  cl.user_session.set("chat_history", chat_history)
  await cl.Message(content=response, author="IskolarBot").send()
