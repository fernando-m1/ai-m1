import streamlit as st

from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain, create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, GCSDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.schema import Document, SystemMessage, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool, Tool
from langchain.vectorstores.faiss import FAISS
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from typing import Any, Iterator, List
import requests
from PIL import Image
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="MoradaUno Chatbot", page_icon="🤖")

# Page title
st.title("MoradaUno Chatbot 🤖")

# Load M1 image
response_img = requests.get("https://github.com/fernando-m1/ai-m1/raw/main/llm-qa/M1-icon-whiteborder.png")
img = Image.open(BytesIO(response_img.content))

# Define functions
def text_splitter():
  return RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n", "####","\n","."],
    chunk_size=1500,
    chunk_overlap=500,
  )

def gcs_loader(bucket, project_name, text_splitter, prefix=None):
    loader = GCSDirectoryLoader(path, prefix, loader_func=UnstructuredMarkdownLoader)
    docs = loader.load_and_split(text_splitter)
    return docs
  
def create_retriever(docs, top_k_results):
  embeddings = OpenAIEmbeddings()
  vectorstore = FAISS.from_documents(docs, embeddings)
  retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_results})
  return retriever

# Load documents
text_splitter = text_splitter()
gcs_project_name = "legal-ai-m1"
gcs_bucket = "moradauno-corpus-demo"

m1_docs = gcs_loader(gcs_bucket, gcs_project_name, text_splitter, prefix='M1_General/')
productos_docs = gcs_loader(gcs_bucket, gcs_project_name, text_splitter, prefix='Productos/')
legal_docs = gcs_loader(gcs_bucket, gcs_project_name, text_splitter, prefix='Legal/')
m1app_docs = gcs_loader(gcs_bucket, gcs_project_name, text_splitter, prefix='M1App/')

# Define chains
chain_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
    Make sure to avoid using any unclear pronouns.
    
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

condense_question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

qa_chain = create_qa_with_sources_chain(llm)

doc_prompt = PromptTemplate(
  template="""<<SYS>> \n Tú nombre es Leyla. Eres un chatbot cuya tarea es responder preguntas sobre MoradaUno.
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas se refieren a MoradaUno.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial.
        Podrás realizar tareas como responder preguntas sobre los servicios de MoradaUno, cotizar los productos que ofrece MoradaUno, y proporcionar documentos con mayor información sobre los servicios de MoradaUno.
        Leyla aprende y mejora constantemente.
        Si te preguntan sobre los servicios de MoradaUno, menciona solamente los principales servicios de MoradaUno, no los servicios adicionales, a menos que te pregunten específicamente por servicios adicionales.
        Leyla no revela ningún otro nombre de empresa bajo ninguna circunstancia.
        Leyla no responde preguntas legales, si el cliente tiene alguna duda legal, Leyla le va a sugerir comunicarlo con un abogado de MoradaUno.
        Leyla siempre debe identificarse como Leyla, asesor de MoradaUno.
        Si se le pide a Leyla que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Leyla, debe responder 'Soy Leyla, un asesor de MoradaUno'.\n <</SYS>> \n\n
        Content: {page_content}\nSource: {source}""",
  input_variables=["page_content", "source"],
)

final_qa_chain = StuffDocumentsChain(
  llm_chain=qa_chain,
  document_variable_name="context",
  document_prompt=doc_prompt,
)

m1_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=m1_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
)

productos_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=productos_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
)

legal_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=legal_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
)

m1app_qa = ConversationalRetrievalChain(
  question_generator=condense_question_chain,
  retriever=m1app_retriever,
  memory=chain_memory,
  combine_docs_chain=final_qa_chain,
)

# Agent setup
system_message = SystemMessage(
    content=(
        """
        Tú nombre es Leyla. Eres un chatbot cuya tarea es responder preguntas sobre MoradaUno.
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas se refieren a MoradaUno.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial".

        Podrás realizar tareas como responder preguntas sobre los servicios de MoradaUno, cotizar los productos que ofrece MoradaUno, y proporcionar documentos con mayor información sobre los servicios de MoradaUno.
        Leyla aprende y mejora constantemente.
        Leyla no revela ningún otro nombre de empresa bajo ninguna circunstancia.
        Leyla no responde preguntas legales, si el cliente tiene alguna duda legal, Leyla le va a sugerir comunicarlo con un abogado de MoradaUno.
        Leyla siempre debe identificarse como Leyla, asesor de MoradaUno.
        Si se le pide a Leyla que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Leyla, debe responder "Soy Leyla, un asesor de MoradaUno".


        TOOLS:
        ------

        Leyla tiene acceso a las siguientes herramientas:
        """
    )
)

chat_history = []

prompt = OpenAIFunctionsAgent.create_prompt(
  system_message=system_message,
  extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history"), MessagesPlaceholder(variable_name="agent_scratchpad")],
)

tools = [
    Tool(
        name="MoradaUno_General_Information_QA_System",
        func=m1_qa.run,
        description="useful for when you need to answer questions at a high-level about MoradaUno. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="MoradaUno_Products_and_Services_QA_System",
        func=productos_qa.run,
        description="useful for when you need to answer questions about MoradaUno's products and services, specially if details and specifications are needed. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="Legal_QA_System",
        func=legal_qa.run,
        description="useful for when you need to answer legal questions. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="M1App_QA_System",
        func=m1app_qa.run,
        description="useful for when you need to answer questions about MoradaUno's M1App, or about a service's procedure. This tool includes a step-by-step guide on how to complete a MoradaUno's tenant screening or rent protection. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
]

agent_memory = AgentTokenBufferMemory(llm=llm) 

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
  agent=agent,
  tools=tools,
  memory=chain_memory,
  verbose=False,
  max_iterations=2,
  early_stopping_method="generate",
  return_intermediate_steps=False,
)

# Streamlit interface
starter_message = "¡Pregúntame sobre MoradaUno! Estoy para resolver tus dudas sobre nuestros servicios."
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar=img).write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    
    # Store the HumanMessage in the session state
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Concatenate history and input
    full_input = "\n".join([msg.content for msg in st.session_state.messages] + [prompt])
    
    response = agent_executor(
        {"input": full_input},
        include_run_info=True,
    )
    response_content = response["output"]
    
    # Escape the $ character
    response_content = response_content.replace("$", "\$")
    
    st.session_state.messages.append(AIMessage(content=response_content))
    st.chat_message("assistant", avatar=img).write(response_content)
