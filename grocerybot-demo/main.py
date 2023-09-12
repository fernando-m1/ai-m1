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
    st.write(f"Loaded {len(docs)} documents.")
    print(f"Loaded {len(docs)} documents.")
    
    doc_chunk = chunks(docs, BATCH_SIZE_EMBEDDINGS)
    st.write(f"Created {len(list(doc_chunk))} chunks.")
    print(f"Created {len(list(doc_chunk))} chunks.")
    
    db = None
    for index, chunk in tqdm(enumerate(doc_chunk)):
        if index == 0:
            db = FAISS.from_documents(chunk, embedding)
        else:
            db.add_documents(chunk)

    # Error handling here
    if db is None:
        raise ValueError("No documents found, unable to create retriever.")

    retriever = db.as_retriever(search_kwargs={"k": top_k_results})
    return retriever

###
def download_github_files_to_temp_dir(api_url: str, folder_name: str) -> str:
    # Make the API request to get the list of files in the GitHub folder
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get files from GitHub: {response.content}")

    tree = json.loads(response.content)['tree']

    # Create a temporary directory to store the downloaded files
    temp_dir = tempfile.mkdtemp()

    # Loop through each file in the GitHub folder
    for file in tree:
        if folder_name in file['path']:
            file_url = file['url']
            file_name = file['path'].split('/')[-1]

            # Download the file content
            file_response = requests.get(file_url)
            if file_response.status_code != 200:
                raise Exception(f"Failed to download file {file_name} from GitHub: {file_response.content}")

            file_content = base64.b64decode(json.loads(file_response.content)['content'])

            # Save the file to the temporary directory
            with open(f'{temp_dir}/{file_name}', 'wb') as f:
                f.write(file_content)

    return temp_dir

# Download 'recipes' folder files into a temporary directory and create the 'recipe_retriever'.
recipes_temp_dir = download_github_files_to_temp_dir(
    api_url="https://api.github.com/repos/fernando-m1/ai-m1/git/trees/main?recursive=1",
    folder_name="recipes/"
)
recipe_retriever = create_retriever(top_k_results=2, dir_path=f"{recipe_temp_dir}/*")

# Download 'products' folder files into a temporary directory and create the 'product_retriever'.
products_temp_dir = download_github_files_to_temp_dir(
    api_url="https://api.github.com/repos/fernando-m1/ai-m1/git/trees/main?recursive=1",
    folder_name="recipes/"
)
product_retriever = create_retriever(top_k_results=2, dir_path=f"{recipe_temp_dir}/*")


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

docs = load_docs_from_directory("./recipes/*")
recipes_detail = {doc.metadata["source"]: doc.page_content for doc in docs}


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
