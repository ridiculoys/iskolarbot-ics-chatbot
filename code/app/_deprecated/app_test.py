import pandas as pd
# from langchain_community.document_loaders import CSVLoader
from langchain.agents.agent_types import AgentType

from ai.agents.pandas.base import create_pandas_dataframe_agent

from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.callbacks import get_openai_callback
from helpers.count_tokens import count_tokens

#for local testing
import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
#for local testing

# for reading csv
# loader = CSVLoader(csv_path)
# docs = loader.load()

# multiselect ?
# options - AI and ML, Data Science, Data Structures, Web Development, Bioinformatics, Web Imaging, General

llm = OpenAI(model="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key, temperature=0, max_tokens=1024)
chat_llm = ChatOpenAI(model="gpt-3.5-turbo-1106", openai_api_key=openai_api_key, temperature=0)

# Read the CSV related to the data, search for a direct answer
def call_csv_agent(csv_path: str, topic: str, question: str):
    # get csv from csv_path
    df = pd.read_csv(csv_path)
    #todo: preprocess data to be in lowercase ?
    
    # Implement logic to search for the answer in the CSV
    #create csv agent to search for answer
    # response should be {'found': 'Yes', answer: 'None'}
    # response should be {'found': 'No', answer: 'Five titles -- find the right format for this'}


    # add before user question
    # **Context from Conversation:**
    #     - Review the conversation history below
    #     {chat_history}
    # - When answering the user's question, make sure to thoroughly consider and analyze the last user question and the last AI answer in the context of the conversation history, emphasizing the most recent exchange
    #     - Refer to earlier parts of the conversation ONLY IF necessary for additional clarification
    #     - If the most recent exchange lacks context, explore earlier parts of the conversation
    #     - Disregard irrelevant conversation and concentrate on the crucial context

    # - Choose the appropriate tools to answer user questions effectively.

    # get different result if answer is found or not
    agent_template = f"""
        You are a professional research virtual assistant who assists researchers in finding answers from a collection of research papers related to {topic}.
        
        **Your role:**
        - Your primary task is to assist users in retrieving the complete and detailed information they need spcifically from the dataset in relation to the csv column headers: {df.columns}
        - If the question is more complex such as asking for a description or for contents within the research papers, give exactly 3 Titles and its respective Authors that might help in the information needed. Use both the keywords and the title to help determine the top 3 most related titles to the user's question. Use the format: ```Title 1 - Author 1 | Title 2 - Author 2 | Title 3 - Author 3```
        
        User Question: {question}
        - You always respond in a helpful, professional, and friendly manner using a friendly conversational tone.
        - You also always ask for clarification if you are unsure about the user's question.
        - If a question cannot be answered using the dataset, respond in a friendly conversational tone that the current data that you have does not contain the information needed to answer the question and ask if there is anything else you can help with.
        
    """

    # Create an agent for performing CSV operations
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=4,
        verbose=True
    )
    print("========= Agent =========")
    print("agent", agent)

    # Run the agent with the given prompt
    with get_openai_callback() as cb:
        # response = agent.run(formatted_prompt)
        response = agent.run({"input": agent_template})
    count_tokens("CSV Agent", "gpt-3.5-turbo-instruct", question, cb)

    return response

# query = input("User Question: ")
# response = call_csv_agent('data/ai.csv', 'Artificial Intelligence', 'Give me a list of papers that were published in 2023')
# response = call_csv_agent('data/ai.csv', 'Artificial Intelligence', 'What is the range of the years in the dataset?')
# print("Response: ", response)

# user chooses a topic
# user asks a question
# specific CSV is called depending on the topic
question = input("User Question: ")
while question != '0':
    question = input("User Question: ")
    response = call_csv_agent('data/ai.csv', 'Artificial Intelligence', question)
    print("Response: ", response)

    if response == "keyword":
        print("Access Vectorstore")

# choose a topic

# if topic chosen, search for answer in csv
    # create csv agent
    # if answer found, return answer
    # else, say "None"
    # prompt: USE TELEGRAPHIC STYLE OF ANSWERING.
# do vectorstore search if not in csv