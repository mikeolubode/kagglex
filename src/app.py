from dotenv import load_dotenv, find_dotenv
from io import BytesIO
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import ConfluenceLoader
from langchain.embeddings import GooglePalmEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms import GooglePalm
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStore
from prompt import confluence_question_prompt, standalone_question_prompt
import PyPDF2
import chainlit as cl
import logging
import os

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)

CONFLUENCE_URL = "https://mikesofts.atlassian.net/wiki"
CONFLUENCE_EMAIL = os.environ["EMAIL"]
CONFLUENCE_API_TOKEN = os.environ["CONFLUENCE_API_TOKEN"]
CONFLUENCE_SPACE_KEY = "~614914d4071141006ab46038"
GOOGLE_PALM_API_KEY = os.environ["GOOGLE_PALM_API_KEY"]


def load_confluence_documents(
    url: str, username: str, space_key: str, CONFLUENCE_API_TOKEN: str, limit: int = -1
):
    """Loads documents from confluence page

    Args:
        url (str): url of the confluence page
        username (str): email/username of the confluence page
        space_key (str): space key of the confluence space to extract
        CONFLUENCE_API_TOKEN (str): api token generated from confluence
        limit (int, optional): Limits the number of pages loaded. Defaults to -1.

    Returns:
        _type_: langchain documents
    """
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
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    )
    texts = text_splitter.split_documents(documents)
    return texts


def split_user_text(user_texts, chunk_size, overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
    )

    texts = text_splitter.split_text(user_texts)
    return texts


def parse_pdf_file(pdf_file):
    pdf_stream = BytesIO(pdf_file.content)
    pdf = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    return pdf_text


async def get_user_vectorstore(
    embeddings: Embeddings, persist_directory: str
) -> VectorStore:
    """_summary_

    Args:
        embeddings (Embeddings): _description_
        texts (List[str]): _description_
        persist_directory (str): _description_

    Returns:
        VectorStore: _description_
    """
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a pdf file to begin!",
            accept={"text/plain": [".pdf"]},
            max_files=1,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    pdf_text = parse_pdf_file(file)

    # Split the text into chunks
    texts = split_user_text(pdf_text, chunk_size=1000, overlap=100)
    vector_db = Chroma.from_texts(texts, embedding=embeddings)

    msg.content = f"{file.name} processed. You may now talk to it!"
    await msg.update()
    return vector_db


async def get_sample_vectorstore(
    embeddings: Embeddings, persist_directory: str
) -> VectorStore:
    """_summary_

    Args:
        embeddings (Embeddings): _description_
        texts (List[str]): _description_
        persist_directory (str): _description_

    Returns:
        VectorStore: _description_
    """
    msg = cl.Message(
        content="User provided no data. Loading sample data...",
        disable_human_feedback=True,
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

    # Check if the folder exists in the current working directory
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        logger.info(f"The folder '{persist_directory}' exists. Loading...")
        vector_db = Chroma(
            embedding_function=embeddings, persist_directory=persist_directory
        )
    else:
        logger.info("No persist directory. Encoding embeddings...")

        vector_db = Chroma.from_documents(
            texts, embedding=embeddings, persist_directory=persist_directory
        )

    msg.content = "Now you can talk to your data! If you didn't upload any data, I have a sample data to keep you company!"
    await msg.update()

    return vector_db


@cl.on_chat_start
async def start():
    """_summary_"""
    persist_directory = "chroma_db"
    embeddings = GooglePalmEmbeddings(google_api_key=GOOGLE_PALM_API_KEY)
    # Load and embed documents to Chroma
    # Wait for the user to upload a file
    res = await cl.AskActionMessage(
        content="Pick Data source!",
        actions=[
            cl.Action(name="Upload", value="user_data", label="upload pdf"),
            cl.Action(name="Sample", value="sample_data", label="sample data"),
        ],
    ).send()

    if res and res.get("value") == "user_data":
        vectorstore = await get_user_vectorstore(embeddings, persist_directory)
    else:
        vectorstore = await get_sample_vectorstore(embeddings, persist_directory)

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
        combine_docs_chain_kwargs={"prompt": standalone_question_prompt},
        memory=memory,
        verbose=False,
        return_source_documents=True,
    )
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    """_summary_

    Args:
        query (str): _description_
    """
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    result = chain({"question": message.content}, callbacks=[cb])

    await cl.Message(content=result["answer"]).send()
