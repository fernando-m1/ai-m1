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

def load_texts_from_loader(loader: Any) -> List[Any]:
    """
    Load documents using the provided loader and split them into texts.
    
    Args:
        loader: The loader object to load documents from.
        
    Returns:
        List of texts obtained by splitting the documents.
    """
    # Load documents
    documents = loader.load()
    
    # Split documents into texts
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    return texts

def create_retriever_from_texts(texts: List[Any], top_k_results: int) -> VectorStoreRetriever:
    """
    Create a retriever using the provided texts and top_k_results.
    
    Args:
        texts: List of texts to create embeddings from.
        top_k_results: Number of top results to retrieve.
        
    Returns:
        A retriever object.
    """
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create FAISS index (or your specific VectorStore)
    docsearch = FAISS.from_documents(texts, embeddings)
    
    # Create a retriever with top k results
    retriever = docsearch.as_retriever(search_kwargs={"k": top_k_results})
    
    return retriever
####

# Load recipes files and create a retriever of recipes.
recipes_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus", prefix="recipes")
recipes_texts = load_texts_from_loader(recipes_loader)
st.write(f"Recipes number of documents: {len(recipes_texts)}")
st.write(f"Recipes texts: {recipes_texts[0]}")
recipes_retriever = create_retriever_from_texts(recipes_texts, 2)
st.write(f"Recipes retriever: {recipes_retriever}")

# Load products files and create a retriever of products.
products_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus", prefix="products")
products_texts = load_texts_from_loader(products_loader)
st.write(f"Products number of documents: {len(products_texts)}")
st.write(f"Products texts: {products_texts[0]}")
products_retriever = create_retriever_from_texts(products_texts, 5)
st.write(f"Products retriever: {products_retriever}")
