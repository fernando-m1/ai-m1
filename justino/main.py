### SETUP: IMPORT LIBRARIES AND SET ENVIRONMENT VARIABLES ###

"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import PythonREPL
from langchain.schema import BaseOutputParser
from langchain.agents.agent import AgentExecutor
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import requests
import urllib.request

# GitHub file URL
#file_url = "https://github.com/fernando-m1/ai-m1/blob/main/justino/credentials.json"

# Send a GET request to download the file
#credentials_json = requests.get(file_url)

# Save the file locally
#with open("credentials.json", "wb") as file:
#    file.write(credentials_json.content)

#AGREGADO PARA IMAGENES
from PIL import Image
urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/logo-tec.png', 'logo_tec_de_monterrey')
image = Image.open('logo_tec_de_monterrey')

urllib.request.urlretrieve('https://raw.githubusercontent.com/a01110946/chatbot/main/tec_de_monterrey/agent-v1.png', 'agent-image')
image2 = Image.open('agent-image')
### FIN DE AGREGADO PARA IMAGENES


# SECCION DE ENCABEZADOS Y PANTALLA DE INICIO
# From here down is all the StreamLit UI.
#st.set_page_config(page_title="Tec de Monterrey - Chatbot", page_icon=":robot:", layout="wide")
with st.container():  
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(image,use_column_width='auto')#despliega logo
        st.header('InfoChat Tec')
        st.markdown("""
                    Podemos ayudarte con todo lo que necesitas saber a cerca de los programas 
                    de estudio en el Tecnológico de Monterrey
                    """)
    with right_column:
        st.image(image2,use_column_width='auto') #despliega imagen
        
##### PRUEBA #####


SYSTEM_MESSAGE = """Assistant es un asesor que responde preguntas del Tec de Monterrey.

Cuando le preguntes algo, te responderá en base a la siguiente información disponible:

El proceso de inscripción para una carrera profesional del Tec de Monterrey es el siguiente:

1. Regístrate y llena tu solicitud de admisión en línea.
2. Integra tu expediente con todo lo que te hace una persona única.
3. Cubre la cuota del proceso.
4. Programa y presenta tus pruebas.
5. Conoce tu resultado de admisión.
6. Inscríbete y forma parte de la comunidad del Tecnológico de Monterrey.

Para revisar de forma especifica tu caso favor de contactar al Tec de Monterrey a través del siguiente link: [https://tec.mx/es/profesional/proceso-de-admision]

El proceso de inscripción para una posgrado del Tec de Monterrey es el siguiente:

1. Contacta a un asesor.
2. Identifícate y llena tu solicitud de admision. 
3. Inicia tu proceso.
4. Prepárate para una entrevista.
5. Prepárate para el examen de admisión al posgrado.
6. Completa tu expediente.
7. Consulta los resultados de adminsion.
8. Incríbete y forma parte del Tec de Monterrey.

Para obtener más detalles de cada uno de los puntos, favor de dirigirse al siguiente link: [https://maestriasydiplomados.tec.mx/admisiones]

Human: {human_input}
Assistant:
"""


prompt = PromptTemplate(input_variables=["human_input"], template=SYSTEM_MESSAGE)


chatgpt_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferMemory(
        memory_key="chat_history", return_messages=True),
)

#### TERMINA PRUEBA ####


st.write("---")
######

st.sidebar.header('Hola, Bienvenido(a)')
st.sidebar.markdown("""
Esta App tiene por objeto contestar a tus dudas sobre las carreras 
profesionales así como los posgrados que tiene el Tec de Monterrey.
    
Realiza la preguntas a nuestro Chatbot.
""")
###### FIN DE PANTALLA DE INICIO


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("Tú: ", "Hola, quiero realizar unos cambios al contrato de arrendamiento, ¿podrías ayudarme?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chatgpt_chain.predict(human_input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")













##### CÓDIGO PARA CARGAR STREAMLIT DE INICIO #####
st.set_page_config(layout="wide", page_title="Colector de Texto | Google Docs", page_icon="📄")
st.title("Colector de Texto | Google Docs")
st.header("Extrae el texto de un Google Doc.")

st.markdown("### **Extrae el texto de un Google Doc utilizando la API de Google Docs, solo especifica el ID del documento.**")
st.markdown("#### ID del documento:")

def get_id():
  input_text = st.text_area(label="", placeholder="Escribe aquí el ID del documento...", key="Hola mundo!")
  return input_text

#doc_id = get_id()

st.markdown("#### El documento contiene el siguiente texto:")
#if doc_id:
#  doc_text = read_structural_elements(doc_content)
#  st.write(doc_text)
st.write("Hola mundo!")
