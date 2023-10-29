# flake8: noqa
from langchain.prompts import ChatPromptTemplate

# This prompt takes in a question and the chat history, and then it
# rephrases the question to be a more contextual question to be
# asked to the retriever.
condence_prompt_template = """Given the following conversation and a follow up question,
 rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
confluence_question_prompt = ChatPromptTemplate.from_template(condence_prompt_template)


standalone_prompt_template = """
System: You are a chatbot that can answer questions about Data Science, Artificial Intelligence or any other context the user provides to you.

Assistant: Use the following pieces of context to answer the question from the user. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Keep the answer as concise as possible. 
Context: <{context}>
Question: <{question}>
Helpful Answer:"""
standalone_question_prompt = ChatPromptTemplate.from_template(
    standalone_prompt_template
)
