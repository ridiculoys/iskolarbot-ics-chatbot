from langchain.prompts import ChatPromptTemplate

class SummaryPrompts:
  def summary_prompt():
    prompt_template = ChatPromptTemplate.from_template("""
      You are an expert research paper summarizer who helps researchers in summarizing the content of relevant research papers related to the topic: {topic}.

      {question}
      =========
      {summaries}
      =========
      """
    )

    return prompt_template