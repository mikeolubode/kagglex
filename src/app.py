import os
import logging
from typing import List

import chainlit as cl
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import ConfluenceLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma

from prompt import confluence_question_prompt


load_dotenv()
logger = logging.getLogger(__name__)


CONFLUENCE_URL = "https://mikesofts.atlassian.net/wiki"
CONFLUENCE_EMAIL = os.environ["EMAIL"]
CONFLUENCE_API_TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
CONFLUENCE_SPACE_KEY = "~614914d4071141006ab46038"
GOOGLE_PALM_API_KEY = os.environ["GOOGLE_PALM_API_KEY"]


def load_confluence_documents(
    url: str, username: str, space_key: str, CONFLUENCE_API_TOKEN: str, limit: int = -1
):
    # get documents from confluence
    loader = ConfluenceLoader(
        url=url,
        username=username,
        api_key=CONFLUENCE_API_TOKEN,
    )
    if limit != -1:
        return loader.load(space_key=space_key, limit=limit)

    return loader.load(space_key=space_key)


def split_document(documents, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
        separators=[" ", "\n", ".", ","],
    )
    texts = text_splitter.split_documents(documents)
    return texts


def get_vectorstore(
    embeddings: Embeddings, texts: List[str], persist_directory: str
) -> VectorStore:
    """
    embeddings: text embedding model instance
    texts: list of texts to embed
    persist_directory: directory to persist vectors
    """
    # Check if the folder exists in the current working directory
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        logger.info(f"The folder '{persist_directory}' exists. Loading...")
        return Chroma(
            embedding_function=embeddings, persist_directory=persist_directory
        )

    logger.info("No persist directory. Encoding embeddings...")
    return Chroma.from_documents(
        texts, embedding=embeddings, persist_directory=persist_directory
    )


@cl.on_chat_start
async def start():
    persist_directory = "chroma_db"

    # Load and embed documents to Chroma
    msg = cl.Message(
        content=f"Getting the data ready. Please wait...", disable_human_feedback=True
    )
    await msg.send()

    documents = load_confluence_documents(
        CONFLUENCE_URL,
        CONFLUENCE_EMAIL,
        CONFLUENCE_SPACE_KEY,
        CONFLUENCE_API_TOKEN,
        limit=-1,
    )
    texts = split_document(documents, chunk_size=1000, overlap=100)

    embeddings = GooglePalmEmbeddings(google_api_key=GOOGLE_PALM_API_KEY)
    vectorstore = get_vectorstore(embeddings, texts, persist_directory)

    # Setup Conversational Retrieval Chain
    llm = GooglePalm(google_api_key=GOOGLE_PALM_API_KEY, temperature=0.1)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=confluence_question_prompt,
        memory=memory,
        verbose=False,
        return_source_documents=True,
    )
    msg.content = "Michael's Confluence chat is ready! Ask me anything!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(query: str):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    result = await chain.acall(query, callbacks=[cb])
    # result = chain({"question": query}, callbacks=[cb])

    await cl.Message(content=result["answer"]).send()
