"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os.path
import os
import requests
import sys
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
import googleapiclient.discovery as discovery
import json

from langchain.chains import ConversationChain
from langchain.llms import OpenAI

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Set Google Docs API environment variables

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/documents.readonly']
DISCOVERY_DOC = 'https://docs.googleapis.com/$discovery/rest?version=v1'

# The Document ID of the contract agreement to work with.
DOCUMENT_ID = '1SwVlU6ZKArnW9pfEQCi5YmaMPr2TkSrseRd-0PQ5Ys0'    # NECESITAMOS CAMBIAR AQUÍ PARA QUE LA BASE DE DATOS ALIMENTE ESTE VALOR

cred_data = {
	"installed": {
		"client_id": st.secrets["CLIENT_ID"],
		"project_id": st.secrets["PROJECT_ID"],
		"auth_uri": st.secrets["AUTH_URI"],
		"token_uri": st.secrets["TOKEN_URI"],
		"auth_provider_x509_cert_url": st.secrets["AUTH_PROVIDER_X509_CERT_URL"],
		"client_secret": st.secrets["CLIENT_SECRETS"],
		"redirect_uris": ["http://localhost"]
	}
}

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
                cred_data, SCOPES)
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

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

chain = load_chain()

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text

st.text(text)

user_input = get_text()

if user_input:
    output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
