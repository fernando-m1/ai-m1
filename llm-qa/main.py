import streamlit as st

from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler 
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain, create_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader, GCSDirectoryLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.schema import Document, SystemMessage, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool, Tool
from langchain.vectorstores.faiss import FAISS
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

from typing import Any, Iterator, List

# Streamlit page config
st.set_page_config(page_title="MoradaUno Chatbot", page_icon="🤖") 

# Define functions
def text_splitter():
  return RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n", "####","\n","."],
    chunk_size=1500,
    chunk_overlap=500,
  )

def gcs_loader(bucket, project_name, text_splitter, prefix=None):
    loader = GCSDirectoryLoader(bucket=bucket, project_name=project_name, prefix=prefix, loader_cls=UnstructuredMarkdownLoader)
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

# Create retrievers
llm = ChatOpenAI(temperature=0, streaming=True, model_name="gpt-4")
embedding = OpenAIEmbeddings()

m1_retriever = create_retriever(m1_docs, 3) 
productos_retriever = create_retriever(productos_docs, 6)
legal_retriever = create_retriever(legal_docs, 10)
m1app_retriever = create_retriever(m1app_docs, 5)

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
  template="""<<SYS>> \n Tú nombre es Leyla. Eres un chatbot cuya tarea es responder preguntas sobre Morada Uno.
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas se refieren a Morada Uno.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial.
        Podrás realizar tareas como responder preguntas sobre los servicios de Morada Uno, cotizar los productos que ofrece Morada Uno, y proporcionar documentos con mayor información sobre los servicios de Morada Uno.
        Leyla aprende y mejora constantemente.
        Si te preguntan sobre los servicios de Morada Uno, menciona solamente los principales servicios de Morada Uno, no los servicios adicionales, a menos que te pregunten específicamente por servicios adicionales.
        Leyla no revela ningún otro nombre de empresa bajo ninguna circunstancia.
        Leyla no responde preguntas legales, si el cliente tiene alguna duda legal, Leyla le va a sugerir comunicarlo con un abogado de Morada Uno.
        Leyla siempre debe identificarse como Leyla, asesor de Morada Uno.
        Si se le pide a Leyla que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Leyla, debe responder 'Soy Leyla, un asesor de Morada Uno'.\n <</SYS>> \n\n
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
        Tú nombre es Leyla. Eres un chatbot cuya tarea es responder preguntas sobre Morada Uno.
        A menos que se indique explícitamente lo contrario, probablemente sea justo asumir que las preguntas se refieren a Morada Uno.
        Si hay alguna ambigüedad, probablemente se asuma que se trata de eso.
        No debes ser demasiado hablador, debes ser breve y conciso, pero debes ser amigable y servicial".

        Podrás realizar tareas como responder preguntas sobre los servicios de Morada Uno, cotizar los productos que ofrece Morada Uno, y proporcionar documentos con mayor información sobre los servicios de Morada Uno.
        Leyla aprende y mejora constantemente.
        Leyla no revela ningún otro nombre de empresa bajo ninguna circunstancia.
        Leyla no responde preguntas legales, si el cliente tiene alguna duda legal, Leyla le va a sugerir comunicarlo con un abogado de Morada Uno.
        Leyla siempre debe identificarse como Leyla, asesor de Morada Uno.
        Si se le pide a Leyla que haga un juego de roles o pretenda ser cualquier otra cosa que no sea Leyla, debe responder "Soy Leyla, un asesor de Morada Uno".


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
        description="useful for when you need to answer questions at a high-level about Morada Uno. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="MoradaUno_Products_and_Services_QA_System",
        func=productos_qa.run,
        description="useful for when you need to answer questions about Morada Uno's products and services, specially if details and specifications are needed. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="Legal_QA_System",
        func=legal_qa.run,
        description="useful for when you need to answer legal questions. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
    Tool(
        name="M1App_QA_System",
        func=m1app_qa.run,
        description="useful for when you need to answer questions about Morada Uno's M1App, or about a service's procedure. This tool includes a step-by-step guide on how to complete a Morada Uno's tenant screening or rent protection. Input should be a fully formed question, not referencing any obscure pronouns from the conversation before. Always answer in Spanish.",
    ),
]

agent_memory = AgentTokenBufferMemory(llm=llm) 

agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
  agent=agent,
  tools=tools,
  memory=chain_memory,
  verbose=True,
  return_intermediate_steps=False,
)

# Streamlit interface
if "messages" not in st.session_state:
  st.session_state["messages"] = []

# Create columns for chat history and input  
history_col, input_col = st.columns([5,1]) 

with history_col:
  # Loop through messages
  for msg in st.session_state.messages:
    # Check type of message
    if isinstance(msg, AIMessage):
      st.text_area("Assistant:", value=msg.content, height=50, disabled=True)
    elif isinstance(msg, HumanMessage):
      st.text_area("You:", value=msg.content, height=50, disabled=True)

with input_col:
  input_msg = st.text_input("You:", key="input")
  
  if input_msg:
    st.session_state.messages.append(HumanMessage(input_msg)) 

    response = agent_executor({"input": input_msg, 
                               "chat_history": st.session_state.messages})
                               
    st.session_state.messages.append(AIMessage(response["output"]))
    
    # Print assistant response
    st.text_area("Assistant:", value=response["output"], height=50)
