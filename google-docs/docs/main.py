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
from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


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

#from dotenv import load_dotenv
#from pyprojroot import here
#import json

SCOPES = 'https://www.googleapis.com/auth/documents.readonly'
DISCOVERY_DOC = 'https://docs.googleapis.com/$discovery/rest?version=v1'
DOCUMENT_ID = '1SwVlU6ZKArnW9pfEQCi5YmaMPr2TkSrseRd-0PQ5Ys0'

##### PRUEBA 02 - GOOGLE API PYTHON QUICKSTART #####

# If modifying these scopes, delete the file token.json.

# The ID of a sample document.
#

def main():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = st.secrets.credentials()
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    '''
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
    '''

    try:
        service = build('docs', 'v1', credentials=creds)

        # Retrieve the documents contents from the Docs service.
        document = service.documents().get(documentId=DOCUMENT_ID).execute()

        print('The title of the document is: {}'.format(document.get('title')))
    except HttpError as err:
        print(err)


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