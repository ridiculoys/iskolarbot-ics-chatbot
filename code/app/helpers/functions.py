import chainlit as cl

#todo: include tables and images as part of the content: https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/LangChain_Multi_modal_RAG.ipynb
#todo: 
from app.helpers.setup import setup_summary_chain, setup_search_content_chain
from app.templates.summary_prompts import SummaryPrompts
async def summarize_paper(vectorstore, args, user_query, topic, index_name):
  try:
    if "focus_on" in args or "author" in args or "summary_length" in args:
      focus_on = f"- specifically about the {args['focus_on']}" if "focus_on" in args else ""
      author = f"The paper authors are {args['paper_authors']}" if "author" in args else ""
      length = f"Strictly limit the result to {args['summary_length']} words" if "summary_length" in args else ""

      to_search = f"{args['paper_title']} {focus_on}"
      results = vectorstore.similarity_search(to_search, k=15)

      summary_template = SummaryPrompts.summary_prompt()

      context = ""
      for idx, doc in enumerate(results):
        content = doc.page_content
        filename = doc.metadata['file_name']

        context += f"""
          Document {idx+1}
          Content: {content}
          Filename: {filename}
          ======
        """
        
      question = f"""
        Summarize the paper entitled "{args['paper_title']}" {focus_on}
        {length}
        {author}
      """
      print(f"user: {user_query}\n===\ntemplate: {question}")
      chain = setup_search_content_chain(vectorstore, template=summary_template)
      results = await chain.ainvoke({"topic": topic, "question": question, "summaries" : context})

      return results["answer"]

    template = f"Paper entitled `{args['paper_title']}`"
    results = vectorstore.similarity_search_with_score(template, k=3)

    filenames = list(set([result[0].metadata['file_name'] for result in results]))

    if len(filenames) == 0:
      results = "The requested paper is not found in my current dataset. Would you like to search about another paper?"
      return results

    results = await setup_summary_chain(index_name=index_name, filename=filenames[0])
  except Exception as e:
    results = f"query failed with error: {e}"
  return results

from app.helpers.setup import setup_search_papers_chain
def get_related_literature(vectorstore, args, user_query, index_name, chat_history):
  try:
    #todo: vectorstore similarity, get metadata
    #prereq: re-upsert documents with proper references

    # args has topic, limit
    response = setup_search_papers_chain(vectorstore=vectorstore, query=user_query, topic=index_name)
  except Exception as e:
    response = f"query failed with error: {e}"
  return response

async def answer_user_query(vectorstore, args, user_query, topic, chat_history):
  try:
    chain = cl.user_session.get("query_chain")

    documents = vectorstore.similarity_search(user_query, k=5)

    context = ""
    for idx, doc in enumerate(documents):
      content = doc.page_content
      filename = doc.metadata['file_name']
      # reference = doc.metadata['reference']

      context += f"""
        Document #{idx+1}
        Content:
          ```{content}```

        Exact Source: `{filename}`
        ==
      """

    # print("context", context)
    # print("chat history", chat_history)
    processed_chat_history = chat_history[1:-1] if len(chat_history) > 3 else ""

    print("processed_chat_history", processed_chat_history) 
    # inputs = {"topic":topic, "question":user_query, "summaries": context, "history": processed_chat_history}
    # chain_response = await chain.ainvoke(inputs)
    # response = chain_response["answer"]

    # args has type_of_question, paper_topic, query_details
    response = "Query chain here"
  except Exception as e:
    response = f"query failed with error: {e}"
  return response

import json
async def execute_function_call(vectorstore, message, user_query, topic, index_name, chat_history):
    if message.tool_calls[0].function.name == "summarize_paper":
      args = json.loads(message.tool_calls[0].function.arguments)
      print("args", args)
      results = await summarize_paper(vectorstore, args, user_query, topic, index_name)
    elif message.tool_calls[0].function.name == "get_related_literature":
      args = json.loads(message.tool_calls[0].function.arguments)
      print("args", args)
      results = get_related_literature(vectorstore, args, user_query, index_name, chat_history)
    elif message.tool_calls[0].function.name == "get_answer": 
      args = json.loads(message.tool_calls[0].function.arguments)
      print("args", args)
      results = await answer_user_query(vectorstore, args, user_query, topic, chat_history)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results