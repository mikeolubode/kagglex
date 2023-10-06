import chainlit as cl
from dotenv import load_dotenv
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# built-in modules
import os
load_dotenv()


def load_confluence_documents(url, username, api_token, limit=-1):
    # get documents from confluence
    loader = ConfluenceLoader(
        url=url, username=username, api_key=api_token, 
    )
    if limit != -1:
        documents = loader.load(space_key=space_key, limit=limit)
    else:
        documents = loader.load(space_key=space_key)
    return documents


def split_document(documents, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )
    texts = text_splitter.split_documents(documents)
    return texts


def get_retriever(embeddings, texts, persist_directory):
    # Check if the folder exists in the current working directory
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(f"The folder '{persist_directory}' exists in the current working directory.")
        cdb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        print(f"The folder '{persist_directory}' does not exist in the current working directory.")
        cdb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=persist_directory)
    
    retriever = cdb.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever


confluence_url = "https://mikesofts.atlassian.net/wiki"
email = os.environ['EMAIL']
api_token = os.environ['API_TOKEN']
space_key = "~614914d4071141006ab46038"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "chroma_db"

documents = load_confluence_documents(confluence_url, email, api_token, limit=-1)
texts = split_document(documents, chunk_size=100, overlap=10)
retriever = get_retriever(embeddings, texts, persist_directory)


@cl.on_message
async def main(question: str):
    documents_results = retriever.get_relevant_documents(question)
    text_elements = [cl.Text(content=doc.metadata['source'], name=doc.metadata['title']) for doc in documents_results]

    source_names = [text_el.name for text_el in text_elements]
    results = "\n\n".join([f"{doc.page_content}: {source}" for doc, source in zip(documents_results, source_names)])

    await cl.Message(
        content=results, elements=text_elements
    ).send()
