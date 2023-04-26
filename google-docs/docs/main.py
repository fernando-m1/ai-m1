# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Recursively extracts the text from a Google Doc.
"""
import googleapiclient.discovery as discovery
from httplib2 import Http
from oauth2client import client
from oauth2client import file
from oauth2client import tools
from httpx_oauth.oauth2 import OAuth2
import streamlit as st
from httpx_oauth.clients.google import GoogleOAuth2
import os
import asyncio

SCOPES = 'https://www.googleapis.com/auth/documents.readonly'
DISCOVERY_DOC = 'https://docs.googleapis.com/$discovery/rest?version=v1'
DOCUMENT_ID = '1SwVlU6ZKArnW9pfEQCi5YmaMPr2TkSrseRd-0PQ5Ys0'


##### PRUEBA 02 - GOOGLE API PYTHON QUICKSTART #####

from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.

# The ID of a sample document.

def main():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
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

    try:
        service = build('docs', 'v1', credentials=creds)

        # Retrieve the documents contents from the Docs service.
        document = service.documents().get(documentId=DOCUMENT_ID).execute()

        print('The title of the document is: {}'.format(document.get('title')))
    except HttpError as err:
        print(err)


if __name__ == '__main__':
    main()

#### AUTENTICACIÓN ####
client_id = os.environ["CLIENT_ID"],
client_secret = os.environ["CLIENT_SECRET"]
redirect_uri = os.environ["REDIRECT_URI"]

client = GoogleOAuth2(client_id, client_secret)

#login_info = OAuth2(
#        client_id = os.environ[CLIENT_ID],
#        client_secret = os.environ[CLIENT_SECRET],
#        redirect_uri = os.environ['REDIRECT_URI'],
#        login_button_text="Continue with Google",
#        logout_button_text="Logout",
#    )

#if login_info:
#        user_id, user_email = login_info
#        st.write(f"Welcome {user_email}")
#else:
#        st.write("Please login")



#### CODIGO DE AUTENTICACIÓN ####

async def write_authorization_url(client,
                                  redirect_uri):
    authorization_url = await client.get_authorization_url(
        redirect_uri,
        scope=["email"],
        extras_params={"access_type": "offline"},
    )
    return authorization_url
authorization_url = asyncio.run(
    write_authorization_url(client=client,
                            redirect_uri=redirect_uri)
)
st.write(f'''<h1>
    Please login using this <a target="_self"
    href="{authorization_url}">url</a></h1>''',
         unsafe_allow_html=True)


st.experimental_get_query_params()

code = st.experimental_get_query_params()       #['code']

async def write_access_token(client,
                             redirect_uri,
                             code):
    token = await client.get_access_token(code, redirect_uri)
    return token
token = asyncio.run(
    write_access_token(client=client,
                       redirect_uri=redirect_uri,
                       code=code))
session_state.token = token



#### INICIA CODIGO DE GOOGLE DOCS API ####

"""
def get_credentials():
    '''Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth 2.0 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    '''
    store = file.Storage('token.json')
    credentials = store.get()

    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        credentials = tools.run_flow(flow, store)
    return credentials
"""

def read_paragraph_element(element):
    """Returns the text in the given ParagraphElement.

        Args:
            element: a ParagraphElement from a Google Doc.
    """
    text_run = element.get('textRun')
    if not text_run:
        return ''
    return text_run.get('content')


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


def main01():
    """Uses the Docs API to print out the text of a document."""
    credentials = get_credentials()
    http = credentials.authorize(Http())
    docs_service = discovery.build(
        'docs', 'v1', http=http, discoveryServiceUrl=DISCOVERY_DOC)
    doc = docs_service.documents().get(documentId=DOCUMENT_ID).execute()
    doc_content = doc.get('body').get('content')
    print(read_structural_elements(doc_content))

if __name__ == '__main01__':
    main01()


##### CÓDIGO PARA CARGAR STREAMLIT DE INICIO #####
st.set_page_config(layout="wide", page_title="Colector de Texto | Google Docs", page_icon="📄")
st.title("Colector de Texto | Google Docs")
st.header("Extrae el texto de un Google Doc.")

st.markdown("### **Extrae el texto de un Google Doc utilizando la API de Google Docs, solo especifica el ID del documento.**")
st.markdown("#### ID del documento:")
def get_id():
  input_text = st.text_area(label="", placeholder="Escribe aquí el ID del documento...", key="doc_id")
  return input_text

doc_id = get_id()

st.markdown("#### El documento contiene el siguiente texto:")
if doc_id:
  doc_text = read_structural_elements(doc_content)
  st.write(doc_text)