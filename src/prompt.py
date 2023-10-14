# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

# This prompt takes in a question and the chat history, and then it
# rephrases the question to be a more contextual question to be
# asked to the retriever.
condence_prompt_template = """Given the following conversation and a follow up question,
 rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
confluence_question_prompt = PromptTemplate.from_template(condence_prompt_template)


# NOTE: not used.
# prompt_template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""
# QA_PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
