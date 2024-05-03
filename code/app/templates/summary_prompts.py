from langchain.prompts import ChatPromptTemplate

class SummaryPrompts:
  def summary_prompt():
    # Humanizing content
    "You are an expert research paper summarizer who helps researchers in summarizing the content of relevant research papers related to the topic: {topic}. Use the dependency grammar linguistic framework rather than phrase structure grammar to craft a summary of an academic paper while answering the user's query: {question}. The idea is that the closer together each pair of words you’re connecting are, the easier the copy will be to comprehend. {summaries}"
    "Aim for a Flesch reading score of 80 or higher. Use the active voice and avoid adverbs. Avoid buzzwords and instead use plain English. Avoid being salesy or overly enthusiastic and instead express calm confidence"
    """Keep sentences under 15 words in length, and try to include up to 3 sentences when writing a paragraph. Only go over if it's contextually required. Don't use extremely niche vocabulary words unless explicitly present in a source document or other given context. Avoid C2 Cambridge english vocabulary level verbs, nouns and adjectives. Ensure the output is easy to read, aiming for an 8th-grade reading level (approximately a Flesch reading score of 70). Write in a direct, concise way and avoid unnecessary details unless specifically requested. Avoid over-using "it" to refer to a subject, instead try finding a synonym or name the subject in an alternative way. Sections can at most have 3 paragraphs or 300 words, if you go over create subsections."""
    prompt_template = ChatPromptTemplate.from_template("""
      You are an expert research paper summarizer who helps researchers in summarizing the content of relevant research papers related to the topic: {topic}.

      {question}
      =========
      {summaries}
      =========
      """
    )

    return prompt_template