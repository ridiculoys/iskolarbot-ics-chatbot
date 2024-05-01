import chainlit as cl

#todo: include tables and images as part of the content: https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/LangChain_Multi_modal_RAG.ipynb
#todo: 
from app.helpers.setup import setup_summary_chain
# from app.templates.summary_prompts import SummaryPrompts
async def summarize_paper(vectorstore, args, user_query, topic, index_name):
  try:
    if "focus_on" in args or "author" in args:
      focus_on = f"- specifically about the {args['focus_on']}" if "focus_on" in args else ""
      author = f"The paper authors are {args['paper_authors']}" if "author" in args else ""

      to_search = f"""
        {args['paper_title']} {focus_on}
        Keywords: {args['semantic_keywords']}
      """

      results = vectorstore.similarity_search(to_search, k=15)

      context = ""
      for idx, doc in enumerate(results):
        content = doc.page_content

        context += f"""
          Document {idx+1}
          Content: {content}
          ======
        """
        
      question = f"""
        Summarize the paper entitled "{args['paper_title']}" {focus_on}
        {author}
      """

      print(f"user: {user_query}\n===\ntemplate: {question}")
      chain = cl.user_session.get("summary_chain")
      results = await chain.ainvoke({"topic": topic, "question": question, "summaries": context})

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
def get_related_literature(vectorstore, args):
  try:
    # args has topic, semantic_keywords
    topic = args['topic']
    semantic_keywords = args['semantic_keywords']
    query=f"""
    topic: {topic}
    keywords: {semantic_keywords}
    """

    response = setup_search_papers_chain(vectorstore=vectorstore, query=query)

    if not response:
      response = "Unfortunately, my dataset is limited and I did not find any related literature to your query. Would you like to ask about another topic?"

  except Exception as e:
    response = f"query failed with error: {e}"
  return response

async def answer_user_query(vectorstore, args, user_query, topic, chat_history):
  try:
    chain = cl.user_session.get("query_chain")

    documents = vectorstore.similarity_search(user_query, k=10)

    context = ""
    for idx, doc in enumerate(documents):
      content = doc.page_content
      reference = doc.metadata['reference']

      context += f"""
        Document #{idx+1}
        Content:
          ```{content}```

        Exact Reference: `{reference}`
        ==
      """

    # print("context", context)
    # print("chat history", chat_history)
    processed_chat_history = chat_history[1:-1] if len(chat_history) > 3 else ""

    question_type = args['question_type']
    question_subject = args['question_subject']
    semantic_keywords = args['semantic_keywords']

    print("processed_chat_history", processed_chat_history) 
    inputs = {"topic":topic, "question":user_query, "summaries": context, "history": processed_chat_history, "question_type": question_type, "question_subject": question_subject, "semantic_keywords": semantic_keywords}
    chain_response = await chain.ainvoke(inputs)
    response = chain_response["answer"]
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
      results = get_related_literature(vectorstore, args, index_name)
    elif message.tool_calls[0].function.name == "get_answer": 
      args = json.loads(message.tool_calls[0].function.arguments)
      print("args", args)
      results = await answer_user_query(vectorstore, args, user_query, topic, chat_history)
    else:
        results = f"Error: function {message.tool_calls[0].function.name} does not exist"
    return results