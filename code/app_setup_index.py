"""
SETTING UP PINECONE INDEX

This script sets up a Pinecone index for the AI chatbot. It uses the Pinecone Python client to create an index and configure it with the necessary settings. The script deletes existing indices and also checks the status of the index to ensure it is ready for use.

Run: `python app_setup_index.py`
"""


import os
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.environ["OPENAI_API_KEY"]
pinecone_api_key=os.environ["PINECONE_API_KEY"]
use_serverless=os.environ["USE_SERVERLESS"]


#Creating an index
from pinecone import Pinecone, ServerlessSpec, PodSpec

def setup_index(index_name):
  # configure client
  print("Configuring client")
  pc = Pinecone(api_key=pinecone_api_key)

  if use_serverless:
    spec = ServerlessSpec(cloud='aws', region='us-west-2')
  else:
    # if not using a starter index, you should specify a pod_type too
    spec = PodSpec()

  print("Setting up index name", index_name) 

  #deletes if exists 
  if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

  import time

  print("Creating index")
  dimension = 1536 #768 or 1536
  pc.create_index(
    name=index_name,
    dimension=dimension,
    metric="cosine",
    spec=spec
  )

  while not pc.describe_index(index_name).status['ready']:
    print("Preparing index...")
    time.sleep(1)

  print("Index ready")
  index = pc.Index(index_name)
  index.describe_index_stats()

#todo: set index name here
""" 
OPTIONS:
  ics-chatbot-ai
  ics-chatbot-security
  ics-chatbot-algorithms
  ics-chatbot-os
  ics-chatbot-hci
  ics-chatbot-general
"""

options = ["ics-chatbot-ai", "ics-chatbot-security", "ics-chatbot-algorithms", "ics-chatbot-os", "ics-chatbot-hci", "ics-chatbot-general"]
index_name=options[5]
setup_index(index_name)