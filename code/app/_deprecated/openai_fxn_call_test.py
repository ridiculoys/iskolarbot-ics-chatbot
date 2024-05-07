import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]


import json
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored 
GPT_MODEL = "gpt-3.5-turbo-0613"
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
def setup_tools(query):
  tools = [
    {
      "type": "function",
      "function": {
        "name": "summarize_paper",
        "description": "Generates a concise summary of a specified journal or research paper. This function extracts the key points, main ideas, and findings from the paper, providing a clear overview that can be used for quick understanding.",
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
              "description": "The desired length of the summary in words. Default is 1000 words or less if not specified."
            },
            "focus_on": {
              "type": "string",
              "description": "A specific aspect, topic, or section of the paper to focus on or highlight for the summary. It could be the 'introduction', 'related literature', 'methodology', 'results', 'conclusion', some other specific topic within the paper, or 'all' for a comprehensive summary. Default is 'all' if not specified."
            },
            "additional_details": {
              "type": "string",
              "description": "Additional details or specific aspects of the topic the user wants to know more about."
            }
          },
          "required": ["paper_title", "focus_on"],
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "get_related_literature",
        "description": "Retrieves a list of related literature for a specified topic within journals or research papers. This function searches for articles, papers, and other scholarly works that are closely related to the provided topic.",
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
            },
            "additional_details": {
              "type": "string",
              "description": "Additional details or specific aspects of the topic the user wants to know more about."
            }
          },
          "required": ["topic"],
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "answer_user_query",
        "description": "Answers the user's query about topics in journals or research papers. This function can handle queries related to the context, definitions, explanations, follow-ups, and more for a specific topic. It is designed to provide detailed answers to support the user's understanding of the subject matter.",
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
  ]

  messages = []
  messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
  messages.append({"role": "user", "content": query})
  chat_response = chat_completion_request(
      messages, tools=tools
  )
  assistant_message = chat_response.choices[0].message
  messages.append(assistant_message)
  

  if assistant_message.tool_calls:
    return assistant_message
  
  return assistant_message.content


user_input = input("Query: ")
response = setup_tools(user_input)
# while user_input != "exit":
#   response = setup_tools(user_input)
#   print("response: ", response.content)
#   print("response: ", response.tool_calls)
#   user_input = input("Query: ")