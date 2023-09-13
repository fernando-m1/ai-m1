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
    text_splitter = CharacterTextSplitter(chunk_size=1800, chunk_overlap=0)
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
    
####

# Load recipes files and create a retriever of recipes.
recipes_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus", prefix="recipes")
recipes_texts = load_texts_from_loader(recipes_loader)
st.write(f"Recipes number of documents: {len(recipes_texts)}")
st.write(f"Recipes texts: {recipes_texts[0]}")
recipe_retriever = create_retriever_from_texts(recipes_texts, 2)
st.write(f"Recipes retriever: {recipe_retriever}")

# Load products files and create a retriever of products.
products_loader = GCSDirectoryLoader(project_name="legal-ai-m1", bucket="moradauno-corpus", prefix="products")
products_texts = load_texts_from_loader(products_loader)
st.write(f"Products number of documents: {len(products_texts)}")
st.write(f"Products texts: {products_texts[0]}")
product_retriever = create_retriever_from_texts(products_texts, 5)
st.write(f"Products retriever: {product_retriever}")

###

## Agent
# Now that you have created the retrievers, it's time to create the Langchain Agent, which will implement a ReAct-like approach.
# An Agent has access to a suite of tools, which you can think of as Python functions that can potentially do anything you equip it with. What makes the Agent setup unique is its ability to **autonomously** decide which tool to call and in which order, based on the user input.

@tool(return_direct=True)
def retrieve_recipes(query: str) -> str:
    """
    Searches the recipe catalog to find recipes for the query.
    Return the output without processing further.
    """
    docs = recipe_retriever.get_relevant_documents(query)

    return (
        f"Select the recipe you would like to explore further about {query}: [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )

@tool(return_direct=True)
def retrieve_products(query: str) -> str:
    """Searches the product catalog to find products for the query.
    Use it when the user asks for the products available for a specific item. For example `Can you show me which onions I can buy?`
    """
    docs = product_retriever.get_relevant_documents(query)
    return (
        f"I found these products about {query}:  [START CALLBACK FRONTEND] "
        + str([doc.metadata for doc in docs])
        + " [END CALLBACK FRONTEND]"
    )

@tool
def recipe_selector(path: str) -> str:
    """
    Use this when the user selects a recipe.
    You will need to respond to the user telling what are the options once a recipe is selected.
    You can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!
    """
    return "Great choice! I can explain what are the ingredients of the recipe, show you the cooking instructions or suggest you which products to buy from the catalog!"

# docs = load_docs_from_directory(f"{recipes_temp_dir}/*")
# print("Sample documents:", docs[:3])
# st.write("Sample documents:", docs[:3])
# recipes_detail = {doc.metadata["source"]: doc.page_content for doc in docs}


@tool
def get_recipe_detail(path: str) -> str:
    """
    Use it to find more information for a specific recipe, such as the ingredients or the cooking steps.
    Use this to find what are the ingredients for a recipe or the cooking steps.

    Example output:
    Ingredients:

    * 1 pound lasagna noodles
    * 1 pound ground beef
    * 1/2 cup chopped onion
    * 2 cloves garlic, minced
    * 2 (28 ounce) cans crushed tomatoes
    * 1 (15 ounce) can tomato sauce
    * 1 teaspoon dried oregano

    Would you like me to show you the suggested products from the catalogue?
    """
    try:
        return recipes_detail[path]
    except KeyError:
        return "Could not find the details for this recipe"

@tool(return_direct=True)
def get_suggested_products_for_recipe(recipe_path: str) -> str:
    """Use this only if the user would like to buy certain products connected to a specific recipe example 'Can you give me the products I can buy for the lasagne?'",

    Args:
        recipe_path: The recipe path.

    Returns:
        A list of products the user might want to buy.
    """
    recipe_to_product_mapping = {
        "./recipes/lasagne.txt": [
            "./products/angus_beef_lean_mince.txt",
            "./products/large_onions.txt",
            "./products/classic_carrots.txt",
            "./products/classic_tomatoes.txt",
        ]
    }

    return (
        "These are some suggested ingredients for your recipe [START CALLBACK FRONTEND] "
        + str(recipe_to_product_mapping[recipe_path])
        + " [END CALLBACK FRONTEND]"
    )




tools = [
    retrieve_recipes,
    retrieve_products,
    get_recipe_detail,
    get_suggested_products_for_recipe,
    recipe_selector,
]

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
