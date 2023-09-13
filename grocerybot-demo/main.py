import streamlit as st

import glob
import pprint
from typing import Any, Iterator, List
import os
import json
import tempfile
import requests
import base64


from langchain.agents import AgentType, initialize_agent
from langchain.document_loaders import TextLoader
from langchain.embeddings import VertexAIEmbeddings
from langchain.llms import VertexAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.tools import tool
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.document_loaders import GCSDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

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

"# GroceryBot Chat 🤖"

llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4")
embedding = OpenAIEmbeddings()

# @st.cache_resource(ttl="1h")

loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus", prefix="recipes")

documents = loader.load()
st.write(f"Created {len(documents)} documents")
st.write(f"Document 1: {documents[0]}")

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
st.write(f"Text 1: {texts[0]}")

embeddings = OpenAIEmbeddings()
docsearch = Chroma.from_documents(texts, embeddings)

