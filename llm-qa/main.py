##### IMPORTAR LIBRERÍAS #####
import streamlit as st
from langchain import OpenAI, VectorDBQA, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.llms import BaseLLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import GoogleDriveLoader

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

from langchain.chains.base import Chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory, 
                                                  ConversationSummaryMemory, 
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.chains.question_answering import load_qa_chain

from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field
from serpapi import GoogleSearch
from langchain.utilities import SerpAPIWrapper
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

#import magic
import os
import nltk
import config
import inspect
import tiktoken
from getpass import getpass
from collections import deque
from typing import Dict, List, Optional, Any

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)




##### CÓDIGO PARA CARGAR STREAMLIT #####
st.set_page_config(layout="wide", page_title="Chatbot M1", page_icon="🤖")
st.title("Chatbot M1")
st.header("Chatbot con acceso a internet")




##### CHATBOT CON ACCESO A INTERNET #####
# Definimos las herramientas que utilizará nuestro chatbot, en este caso solo utilizará SerpAPIWrapper para realizar búsquedas en Google. 
Google_search = SerpAPIWrapper()
toolsSERP = [
    Tool(
        name = "Current Search",
        func=Google_search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

# Definimos el modelo de lenguaje natural que utilizará nuestro chatbot.
llmSERP=ChatOpenAI(temperature=0,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    model_name="gpt-3.5-turbo"
)

# Definimos la cadena de herramientas que utilizará nuestro chatbot. Los elementos clave son tools y AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
#que le permite al chatbot mantener una conversación a la par que utiliza herramientas .
agent_chain = initialize_agent(toolsSERP, llmSERP, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=ConversationBufferMemory(memory_key="chat_history"))
