### SETUP: IMPORT LIBRARIES AND SET ENVIRONMENT VARIABLES ###

import os.path
import os
import requests
import sys
import streamlit as st

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery as discovery

import re
from typing import Dict, List, Optional, Tuple, Union, Any


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Set Google Docs API environment variables

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
DISCOVERY_DOC = 'https://docs.googleapis.com/$discovery/rest?version=v1'

# The Document ID of the contract agreement to work with.
DOCUMENT_ID = '1SwVlU6ZKArnW9pfEQCi5YmaMPr2TkSrseRd-0PQ5Ys0'    # NECESITAMOS CAMBIAR AQUÍ PARA QUE LA BASE DE DATOS ALIMENTE ESTE VALOR
AGREEMENT_ID = 'A5280'                                          # NECESITAMOS CAMBIAR AQUÍ PARA QUE LA BASE DE DATOS ALIMENTE ESTE VALOR

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']


### GOOGLE DOCS TEXT ###

# Define credentials to access the Google Docs API
def get_credentials():
    
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get('textRun')
    if not text_run:
        return ''
    return text_run.get('content')

def read_paragraph_element_style(element):
    """Returns the text style in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get('textRun')
    if not text_run:
        return ''
    textRun = text_run.get('textRun')
    textStyle = textRun.get('textStyle')
    return textStyle

def read_structural_elements(elements):
    """Recurses through a list of Structural Elements to read a document's text where text may be
        in nested elements.

        Args:
            elements: a list of Structural Elements.
    """
    text = ''
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                text += read_paragraph_element(elem)
        elif 'table' in value:
            # The text in table cells are in nested Structural Elements and tables may be
            # nested.
            table = value.get('table')
            for row in table.get('tableRows'):
                cells = row.get('tableCells')
                for cell in cells:
                    text += read_structural_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text in the TOC is also in a Structural Element.
            toc = value.get('tableOfContents')
            text += read_structural_elements(toc.get('content'))
    return text

def extract_text_elements(elements):
    """Recurses through a list of Structural Elements to extract text elements.

        Args:
            elements: a list of Structural Elements.
    """
    text_elements = []
    for value in elements:
        if 'paragraph' in value:
            elements = value.get('paragraph').get('elements')
            for elem in elements:
                if 'textRun' in elem:
                    text_element = {
                        'startIndex': elem.get('startIndex'),
                        'endIndex': elem.get('endIndex'),
                        'content': elem.get('textRun').get('content'),
                        'textStyle': elem.get('textRun').get('textStyle'),
                    }
                    text_elements.append(text_element)
        elif 'table' in value:
            # The text elements in table cells are in nested Structural Elements and tables may be
            # nested.
            table = value.get('table')
            for row in table.get('tableRows'):
                cells = row.get('tableCells')
                for cell in cells:
                    text_elements += extract_text_elements(cell.get('content'))
        elif 'tableOfContents' in value:
            # The text elements in the TOC are also in a Structural Element.
            toc = value.get('tableOfContents')
            text_elements += extract_text_elements(toc.get('content'))
    return text_elements

# Extract the complete text from the specified Google Doc.
def read_gdocs_style():
    """Uses the Docs API to print out the text of a document."""
    credentials = get_credentials()
    #http = credentials.authorize(Http())
    docs_service = discovery.build(
        'docs', 'v1', credentials=credentials, discoveryServiceUrl=DISCOVERY_DOC)
    doc = docs_service.documents().get(documentId=DOCUMENT_ID).execute()
    doc_content = doc.get('body').get('content')
    #textStyle = extract_text_style(doc_content)
    return doc_content

doc_content = read_gdocs_style()

doc_elements = extract_text_elements(doc_content)

text = read_structural_elements(doc_content)

# Return the complete agreement text, but with formatted text marked up with <BOLD>, <ITAL>, <UNDR> and <BACK> tags.
def formattedText():
    formattedText = ''
    for item in doc_elements:
        content = item['content']
        text_style = item['textStyle']
        if 'bold' in text_style and text_style['bold'] and content != '\n':
            content = '<BOLD>' + content + '</BOLD>'
        if 'italic' in text_style and text_style['italic'] and content != '\n':
            content = '<ITAL>' + content + '</ITAL>'
        if 'underline' in text_style and text_style['underline'] and content != '\n':
            content = '<UNDR>' + content + '</UNDR>'
        if 'backgroundColor' in text_style and text_style['backgroundColor'] and content != '\n':
            content = '<BACK>' + content + '</BACK>'
        formattedText += content
    return formattedText

formattedText = formattedText(doc_elements)



### PROCESSING TEXT ###

user_request = input()

# Define the chat class and specify the LLM model to use.
Chat = ChatOpenAI(verbose=True, model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY, request_timeout=120, max_retries=2)

# Create a schema for the response of the LLM model to get structured data from the response.
response_schemas = [
ResponseSchema(name="friendlyResponse", description="Es la respuesta amigable e introductoria del bot al usuario para presentar el texto original y el texto generado."),
ResponseSchema(name="existingText", description="Es el texto original que está siendo modificado. No debes modificarlo, ni su puntuación, ni gramática, nada."),
ResponseSchema(name="updateText", description="Es el nuevo texto generado por el modelo de lenguaje siguiendo las instrucciones del usuario."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

def LLM_call(contract_text: str, user_request: str) -> Dict[str, str]:
    # Define the template for the LLM call. Send over the complete text of the Google Doc, the user's request, and the format instructions.
    template="""
        Eres un asistente legal del equipo de Morada Uno. Tus usuarios son asesores inmobiliarios gestionando contratos de arrendamiento de sus clientes (arrendador y/o arrendatario).
        Cuando el asesor te solicite realizar cambios al contrato de arrendamiento, debes asumir que se refiere al contrato de arrendamiento que se encuentra entre ---.
        Si dentro del texto del contrato encuentras el marcador <BOLD>, debes mantenerlo cuando generes el nuevo texto.
        El texto del contrato no debes modificarlo, ni su puntuación, ni gramática, nada.

    ---    
    {contract_text}
    ---
    {format_instructions}

    Ejemplo:
    Instrucción 1: "El teléfono del arrendatario debe ser +523310203040"
    Entrada 1: "su número telefónico es: <BOLD>523929270689</BOLD>"
    Salida 1: "su número telefónico es: <BOLD>+523310203040</BOLD>"

    Instrucción 2: "Cambia la pena convencional a un monto equivalente a dos meses del valor de renta."
    Entrada 2: "<BOLD>DÉCIMA TERCERA</BOLD>. <BOLD>Pena Convencional. </BOLD>En caso de que alguna de las \
    Partes incumpla con cualquiera de las obligaciones a su cargo o termine de manera anticipada la relación \
    contractual, estará obligado<BOLD> </BOLD>a pagar por concepto de <BOLD>pena convencional</BOLD>, la cantidad total \
    de <BOLD>$181,768.00 M.N.</BOLD> (Ciento ochenta y uno mil setecientos sesenta y ocho pesos 00/100 M.N.) (la \
    “<BOLD>Pena Convencional</BOLD>”)."
    Salida 2: "<BOLD>DÉCIMA TERCERA</BOLD>. <BOLD>Pena Convencional. </BOLD>En caso de que alguna de las \
    Partes incumpla con cualquiera de las obligaciones a su cargo o termine de manera anticipada la relación \
    contractual, estará obligado<BOLD> </BOLD>a pagar por concepto de <BOLD>pena convencional</BOLD>, la cantidad total \
    de <BOLD>$400,000.00 M.N.</BOLD> (Cuatrocientos mil pesos 00/100 M.N.) (la \
    “<BOLD>Pena Convencional</BOLD>”)."
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{user_request}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    result = Chat(chat_prompt.format_prompt(contract_text=formattedText, user_request=user_request, format_instructions=format_instructions).to_messages())
    dict = output_parser.parse(result.content)

    return dict

dict = LLM_call(formattedText, user_request)

# Clean existingText, removing <BOLD>, <ITAL>, <UNDR> and <BACK> tags. This cleaned text will be used as the containsText within the first replaceAllText request.
clean_existingText = re.sub(r'<(\/?(BOLD|UNDR|ITAL|BACK))>', '', dict['existingText'])

# Find the substrings within each pair of tags on the updateText. These substrings will be used as the replaceText within the first replaceAllText request.
pattern = r'>([^<]*?)</'
matches = re.findall(pattern, dict['updateText'])
tagged_texts = [match for match in matches if match]

# Find the format tags for each tagged text. These tags will be used as the textStyles within the second updateTextStyle request.
tag_labels = []

for item in tagged_texts:
    item_pattern = re.escape(item) + "(</[A-Za-z]{4}>)+"
    match = re.search(item_pattern, dict['updateText'])
    if match:
        tags = re.findall("</[A-Za-z]{4}>", match.group())
        tag_labels.append({'content': item, 'tag': tags})

# Find the index of the first character of the first match in the document. This will be used as the startIndex within the first updateTextStyle request to turn all paragraph to unbolded text.
matching_dict = next((d for d in doc_elements if d.get('content') == tagged_texts[0]), None)
paragraph_start_index = matching_dict['startIndex']

# Find the index of the last character of the updateText text. This will be used as the endIndex within the first updateTextStyle request to turn all paragraph to unbolded text.
paragraph_end_index = paragraph_start_index + len(dict['updateText'])

# Find the relative positions of the tagged texts in the updateText (startIndex, endIndex, length)
def absolute_positions(updateText: str, paragraph_start_index: int):
    relative_positions = []
    start_index = 0

    for match in tagged_texts:
        start_index = dict['updateText'].find(match, start_index)
        end_index = start_index + len(match)
        relative_positions.append([start_index, end_index, len(match)])

    adjusted_paragraph_start_index = paragraph_start_index - relative_positions[0][0] + 6

    # Find the absolute positions of the tagged texts in the updateText (startIndex, endIndex, length) by adding the paragraph_start_index to the relative positions.  
    absolute_positions = []
    for position in relative_positions:
        absolute_positions.append([position[0] + adjusted_paragraph_start_index, position[1] + adjusted_paragraph_start_index + 1, position[2]])
    #print(absolute_positions)       # This will be used to create the range values for the second updateTextStyle request to turn the bolded text to bolded text.

    return absolute_positions

absolute_positions = absolute_positions(dict['updateText'], paragraph_start_index)

# Define the mapping to convert the format tags within the updateText to textStyles keys for the request.
tag_to_text_style_mapping = {
    '</BOLD>': ('bold', True),
    '</ITAL>': ('italic', True),
    '</UNDR>': ('underline', True),
    '</BACK>': ('backgroundColor', {'color': {'rgbColor': {'red': 1, 'green': 0.9490196, 'blue': 0.8}}})
}

# Dynamically generate the requests to be sent to the Google Docs API. One updateTextStyle request per tagged text in the updateText.
# Dynamically place the startIndex and endIndex values, as well as each format key for each request previously calculated in the absolute_positions list.
def dynamic_updateTextStyle_requests(matches: List[str], absolute_positions: List[List[int]], tag_labels: List[Dict[str, str]]):

    requests_list = []
    for match in matches:
        request_json = {
            'updateTextStyle': {
                'range': {'startIndex': None, 'endIndex': None},
                'textStyle': {},
                'fields': "bold, italic, underline, backgroundColor"
            }
        }
        requests_list.append(request_json)

    for i, match in enumerate(matches):
        requests_list[i]['updateTextStyle']['range']['startIndex'] = absolute_positions[i][0]
        requests_list[i]['updateTextStyle']['range']['endIndex'] = absolute_positions[i][1]
        
        # Map the tags to text styles
        for tag in tag_labels[i]['tag']:
            style, value = tag_to_text_style_mapping.get(tag, (None, None))
            if style is not None:
                requests_list[i]['updateTextStyle']['textStyle'][style] = value

    return requests_list

requests_list = dynamic_updateTextStyle_requests(tagged_texts, absolute_positions, tag_labels)



### FORMAT HTTP REQUEST ###

# Sets the necessary configuration for the HTTP request to be sent to the Google Docs API.
Refreshtoken = ""
params = {
       "grant_type": "refresh_token",
       "client_id": "51584486703-onlp0g2c8islv56ap6ukhrc3tiu9ettl.apps.googleusercontent.com",
       "client_secret": "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",
       "refresh_token": "1//04voG16Bb-JDECgYIARAAGAQSNwF-L9IrigZKzDipJBlbrGQhnNgyTiKxsag0fml7DYsLXLib2N0JAtvlWm1-rfM0bjxZ9mC3G6k"
       }

authorization_url = "https://www.googleapis.com/oauth2/v4/token"

r = requests.post(authorization_url, data=params)

if r.ok:
   token = str((r.json()['access_token']))
   Refreshtoken = Refreshtoken + token
else:
  print("Failed")
  sys.exit()

# Define headers for the HTTP request
headers = {
   "Authorization": "Bearer " + str(Refreshtoken),
}

# Define JSON body for the HTTP request
def replaceAllText_json(containsText: str, replaceText: Dict[str, str], paragraph_start_index: int, paragraph_end_index: int) -> Dict[str, str]:
    body_json = {
        "requests": [
            {
                "replaceAllText": {
                "containsText": {
                    "text": containsText,
                    "matchCase": False
                },
                "replaceText": replaceText
                }
            },
            {
                "updateTextStyle": {
                    "range": {
                        "startIndex":paragraph_start_index,
                        "endIndex":paragraph_end_index
                    },
                    "textStyle": {
                        "bold": False
                    },
                    "fields": "bold"
                }
            }
        ]
    }

    return body_json

body = replaceAllText_json(clean_existingText, dict['updateText'], paragraph_start_index, paragraph_end_index)

def make_changes_request(headers: Dict[str, str], body: Dict[str, str], requests_list: List[Dict[str, str]]):
    # replaceAllText request sends the updateText to the Google Docs, and the updateTextStyle request turns the whole paragraph to unbolded text.
    r1 = requests.post(
    url = "https://docs.googleapis.com/v1/documents/" + DOCUMENT_ID + ":batchUpdate",
    headers = headers,
    json = body
    )
    
    # updateTextStyle request turns the text within the tags to their corresponding text style format.
    r2 = requests.post(
    url = "https://docs.googleapis.com/v1/documents/" + DOCUMENT_ID + ":batchUpdate",
    headers = headers,
    json = {'requests': requests_list}
    )

    # replaceAllText request deletes the <BOLD>, </BOLD>, <ITAL>, </ITAL>, <UNDR>, </UNDR>, <BACK> and </BACK>, tags from the document.
    r3 = requests.post(
    url = "https://docs.googleapis.com/v1/documents/" + DOCUMENT_ID + ":batchUpdate",
    headers = headers,
    json = {'requests': [
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '<BOLD>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '</BOLD>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '<ITAL>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '</ITAL>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '<UNDR>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '</UNDR>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '<BACK>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            {
                                'replaceAllText': {
                                    'containsText': {
                                        'text': '</BACK>', 'matchCase': True
                                    },
                                    'replaceText': ''
                                }
                            },
                            
                        ]
            }
    )
    print('¡Listo! He realizado los cambios en el contrato de arrendamiento.')
    return r1.status_code, r2.status_code, r3.status_code



### DEFINE MAIN ###
def main():

    #doc_content = read_gdocs_style()
    
    #doc_elements = extract_text_elements(doc_content)
    
    #formattedText = formattedText(doc_elements)

    #user_request = input()

    #dict = LLM_call(formattedText, user_request)

    #absolute_positions = absolute_positions(dict['updateText'], paragraph_start_index)

    #requests_list = dynamic_updateTextStyle_requests(tagged_texts, absolute_positions, tag_labels)

    #body = replaceAllText_json(clean_existingText, dict['updateText'], paragraph_start_index, paragraph_end_index)

    #make_changes_request(headers, body, requests_list)



### EXECUTE ###
if __name__ == '__main__':
    main()



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
