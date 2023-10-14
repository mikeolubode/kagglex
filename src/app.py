import chainlit as cl
from dotenv import load_dotenv
from langchain.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.chains import ConversationalRetrievalChain


# built-in modules
import os

load_dotenv()


def load_confluence_documents(url, username, CONFLUENCE_API_TOKEN, limit=-1):
    # get documents from confluence
    loader = ConfluenceLoader(
        url=url,
        username=username,
        api_key=CONFLUENCE_API_TOKEN,
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
        separators=[" ", "\n", ".", ","],
    )
    texts = text_splitter.split_documents(documents)
    return texts


def get_vectorstore(embeddings, texts, persist_directory):
    """
    embeddings: text embedding model instance
    texts: list of texts to embed
    persist_directory: directory to persist vectors
    """
    # Check if the folder exists in the current working directory
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(
            f"The folder '{persist_directory}' exists in the current working directory. Hence embeddings will be loaded from disk"
        )
        cdb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    else:
        print(
            f"The folder '{persist_directory}' does not exist in the current working directory. Hence embeddings will be created afresh"
        )
        cdb = Chroma.from_documents(
            texts, embedding=embeddings, persist_directory=persist_directory
        )
    return cdb


confluence_url = "https://mikesofts.atlassian.net/wiki"
email = os.environ["EMAIL"]
CONFLUENCE_API_TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
space_key = "~614914d4071141006ab46038"

embeddings = GooglePalmEmbeddings(google_api_key=os.environ["GOOGLE_PALM_API_KEY"])
googlepalm_llm = GooglePalm(
    google_api_key=os.environ["GOOGLE_PALM_API_KEY"], temperature=0.1
)

persist_directory = "chroma_db"

documents = load_confluence_documents(
    confluence_url, email, CONFLUENCE_API_TOKEN, limit=-1
)
texts = split_document(documents, chunk_size=1000, overlap=100)
vectorstore = get_vectorstore(embeddings, texts, persist_directory)

qa_chain = ConversationalRetrievalChain.from_llm(
    googlepalm_llm,
    retriever=vectorstore.as_retriever(),
    verbose=False,
    return_source_documents=True,
)
chat_history = []


@cl.on_message
async def main(query: str):
    result = qa_chain({"question": query, "chat_history": chat_history})
    await cl.Message(content=result["answer"]).send()
    chat_history.append((query, result["answer"]))
