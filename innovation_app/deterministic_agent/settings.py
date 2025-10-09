import os
import logging
import streamlit as st
#from dotenv import load_dotenv
#load_dotenv('.env',override=True) 

logging.basicConfig(level=logging.ERROR) 

#MODEL = os.getenv('MODEL_MISTRAL') 
#APIKEY = os.getenv('LITELLM_APIKEY')
#BASEURL = os.getenv('LITELLM_URL') 

MODEL = st.secrets['MODEL'] 
APIKEY = st.secrets['APIKEY']
BASEURL = st.secrets['BASEURL'] 

MESSAGE_HISTORY_KEY = "messages_final_mem_v2"



