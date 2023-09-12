import streamlit as st

import glob
import pprint
from typing import Any, Iterator, List

from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm
import vertexai

from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder

st.set_page_config(
    page_title="Morada Uno",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

"# Chat 🤖"

PROJECT_ID = "legal-ai-m1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location="us-central1")

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=256,
    temperature=0,
    top_p=0.8,
    top_k=40,
)

embedding = VertexAIEmbeddings()

@st.cache_resource(ttl="1h")
def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive n-sized chunks from lst.

    Args:
        lst: The list to be chunked.
        n: The size of each chunk.

    Yields:
        A list of the next n elements from lst.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_docs_from_directory(dir_path: str) -> List[Document]:
    """Loads a series of docs from a directory.

    Args:
      dir_path: The path to the directory containing the docs.

    Returns:
      A list of the docs in the directory.
    """

    docs = []
    for file_path in glob.glob(dir_path):
        loader = TextLoader(file_path)
        docs = docs + loader.load()
    return docs


def create_retriever(top_k_results: int, dir_path: str) -> VectorStoreRetriever:
    """Create a recipe retriever from a list of top results and a list of web pages.

    Args:
        top_k_results: number of results to return when retrieving
        dir_path: List of web pages.

    Returns:
        A recipe retriever.
    """

    BATCH_SIZE_EMBEDDINGS = 5
    docs = load_docs_from_directory(dir_path=dir_path)
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    for index, chunk in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)

    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

recipe_retriever = create_retriever(top_k_results=2, dir_path="./recipes/*")
product_retriever = create_retriever(top_k_results=5, dir_path="./products/*")




tool = create_retriever_tool(
    configure_retriever(),
    "search_moradauno_info",
    "Searches and returns documents regarding Morada Uno. Morada Uno is a Mexican technology startup, which has the mission of empowering real estate professionals to close faster and safer transactions. You do not know anything about Morada Uno, so if you are ever asked about Morada Uno you should use this tool.",
)
tools = [tool]
llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
message = SystemMessage(
    content=(
        "You are a helpful chatbot who is tasked with answering questions about Morada Uno. "
        "Unless otherwise explicitly stated, it is probably fair to assume that questions are about Morada Uno. "
        "If there is any ambiguity, you probably assume they are about that."
    )
)
prompt = OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)
memory = AgentTokenBufferMemory(llm=llm)
starter_message = "¡Pregúntame sobre Morada Uno! Estoy para resolver tus dudas sobre nuestros servicios."
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]


for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)


if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_executor(
            {"input": prompt, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input": prompt}, response)
        st.session_state["messages"] = memory.buffer
        run_id = response["__run"].run_id
