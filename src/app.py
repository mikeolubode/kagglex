import chainlit as cl
from dotenv import load_dotenv
from langchain.document_loaders import ConfluenceLoader
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# built-in modules
import requests
import os
import json

confluence_url = "https://mikesofts.atlassian.net/wiki"
load_dotenv()
email = os.environ['EMAIL']
api_token = os.environ['API_TOKEN']

def initialize_retriever():
    ## get documents from confluence
    loader = ConfluenceLoader(
        url=confluence_url, username=email, api_key=api_token, 
    )

    documents = loader.load(space_key="~614914d4071141006ab46038", limit=50)

    ## split documents
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 50,
        chunk_overlap  = 10,
        length_function = len,
        add_start_index = True,
    )
    texts = text_splitter.split_documents(documents)

    # Get embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Create a retriever
    retriever = Chroma.from_documents(texts, embedding=embeddings).as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
retriever = initialize_retriever()
@cl.on_message
async def main(question: str):
    # Your custom logic goes here...
    results = retriever.get_relevant_documents(question)
    # Send a response back to the user
    await cl.Message(
        content=results,
    ).send()