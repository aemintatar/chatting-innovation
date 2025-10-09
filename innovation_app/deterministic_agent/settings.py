import os
import logging
import streamlit as st

# Suppress most ADK internal logs to keep the console clean during Streamlit runs.
# You can change this to logging.INFO or logging.DEBUG for more verbose output during debugging.
logging.basicConfig(level=logging.ERROR) 

MODEL = st.secrets['MODEL'] # Specifies the Google Gemini model to be used by the ADK agent.

APIKEY = st.secrets['APIKEY']

BASEURL = st.secrets['BASEURL'] 

APP_NAME_FOR_ADK = "Innovation Chatbot" # A unique name for your application within ADK, used for session management.

USER_ID = "emin" # A default user ID. In a real application, this would be dynamic (e.g., from a login system).
# Defines the initial state for new ADK sessions. This provides default values for user information.

INITIAL_STATE = {
    "user_name": "Emin",
    "user_hobbies": "",
    "user_interests": ""
}

MESSAGE_HISTORY_KEY = "messages_final_mem_v2" # Key used by Streamlit to store the chat history in its session state.

ADK_SESSION_KEY = "adk_session_id" # Key used by Streamlit to store the unique ADK session ID.

META_ALL_INDEX_KEY = "meta_all_index"

FAISS_TECH_INDEX_KEY = "faiss_tech_index"

META_TECH_INDEX_KEY = "meta_tech_index"

FAISS_SERVICE_INDEX_KEY = "faiss_service_index"

META_SERVICE_INDEX_KEY = "meta_service_index"

FAISS_GOOD_INDEX_KEY = "faiss_good_index"

META_GOOD_INDEX_KEY = "meta_good_index"

FAISS_TECH_LQ_INDEX_KEY = "faiss_tech_lq_index"

META_TECH_LQ_INDEX_KEY = "meta_tech_lq_index"

FAISS_MARKET_LQ_INDEX_KEY = "faiss_market_lq_index"

META_MARKET_LQ_INDEX_KEY = "meta_market_lq_index"

FAISS_DISTANCE_INDEX_KEY = "faiss_distance_index"

META_NUTS2_INDEX_KEY = "meta_nuts2_index"


