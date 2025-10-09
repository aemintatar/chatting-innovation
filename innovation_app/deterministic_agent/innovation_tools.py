from typing import Dict, Any, Optional

from settings import *
from openai import OpenAI
import faiss
import os
import io
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.stats import rankdata
import pandas as pd

# Auxiliary Tools
client = OpenAI(base_url=BASEURL, api_key=APIKEY)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device='cpu')

def load_index(index_path: str=None):
    """Loads a FAISS index and optional metadata."""
    index = None
    if index_path:
        index = faiss.read_index(index_path)
    
    return index

def load_meta(metadata_path: str = None):
    """Loads a FAISS index and optional metadata."""
    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        import pickle
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
    
    return metadata

def load_country_regions(metadata):
    """Load country-region mappings dynamically."""
    country_regions = {}
    for region in metadata:
        country = region.get("Country code")
        label = region.get("NUTS label")
        if country and label:
            country_regions.setdefault(country, []).append(label)
    return country_regions
    
def retrieve_documents_with_location(region_code,topic,query):
    '''
    Using the query, retrieve the relevant documents.
    Assumes location is present. Using locations LQ values filter the relevant documents.
    '''
    #debugg
    print('Starting the function with location')
    #load data based on the topic
    if topic.lower() == "technology":
        faiss_index = st.session_state.get(FAISS_TECH_INDEX_KEY)
        metadata = st.session_state.get(META_TECH_INDEX_KEY)
        lq_metadata = st.session_state.get(META_TECH_LQ_INDEX_KEY)
        lq_variable = 'tech_lq'
        lq_code_variable = 'cpc'
        code_variable = 'CPC_4digit'
        label_variable = 'cpc_4digit_label'
    elif topic.lower() =='service':
        faiss_index = st.session_state.get(FAISS_SERVICE_INDEX_KEY)
        metadata = st.session_state.get(META_SERVICE_INDEX_KEY)
        lq_metadata = st.session_state.get(META_MARKET_LQ_INDEX_KEY)
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        code_variable = 'Nice subclass'
        label_variable = 'Nice_subclass_label'
    elif topic.lower() =='good':
        faiss_index = st.session_state.get(FAISS_GOOD_INDEX_KEY)
        metadata = st.session_state.get(META_GOOD_INDEX_KEY)
        lq_metadata = st.session_state.get(META_MARKET_LQ_INDEX_KEY)
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        code_variable = 'Nice subclass'
        label_variable = 'Nice_subclass_label'
    else:
        return {"status": "error", "message": f"Unsupported topic: {topic}"} 
    
    query_emb = embedding_model.encode([query],convert_to_numpy=True)  # your embedding function
    D, I = faiss_index.search(query_emb, len(metadata))
    for i in I[0]:
        metadata[i]['similarity'] = D[0][i]
    metadata = [metadata[i] for i in I[0]] # this orders from best match to worst
    metadata = pd.DataFrame(metadata)
    print(metadata.head())

    #select based on the LQ scores the codes
    lq_results = [meta for meta in lq_metadata if meta.get('nuts2_code') == region_code]
    lq_results = pd.DataFrame(lq_results)
    lq_results = lq_results.drop(columns = ['nuts2_code','country','country_code']).rename(columns={'country_en':'country','nuts2':'region'})
    
    lq_results = lq_results.merge(right=metadata,how='inner',on='Nice_subclass')
    lq_results = lq_results.sort_values(lq_variable)
    #selected_codes = list(lq_results[lq_code_variable])
    #print(selected_codes)
    #st.session_state['selected_codes'] = selected_codes
    lq_results = lq_results.to_dict(orient="records")[:5]
    print(lq_results)
    return lq_results

def retrieve_documents_without_location(topic,query):
    '''
    Using the query, retrieve the relevant documents.
    '''
    #debugg
    print('Starting the function without location')
    print(query)
    #query the doc fais index
    if topic.lower() == 'technology':
        faiss_index = st.session_state.get(FAISS_TECH_INDEX_KEY)
        metadata = st.session_state.get(META_TECH_INDEX_KEY)
    elif topic.lower() =='service': #bigger than 34
        faiss_index = st.session_state.get(FAISS_SERVICE_INDEX_KEY)
        metadata = st.session_state.get(META_SERVICE_INDEX_KEY)
    elif topic.lower() =='good': #less than 34
        faiss_index = st.session_state.get(FAISS_GOOD_INDEX_KEY)
        metadata = st.session_state.get(META_GOOD_INDEX_KEY)

    query_emb = embedding_model.encode([query],convert_to_numpy=True)  # your embedding function
    D, I = faiss_index.search(query_emb, 5)

    # Retrieve documents
    results = [metadata[i] for i in I[0]]
    for meta in results:
        meta['Nice_subclass_keyword'] = meta['Nice_subclass_keyword'].replace('|',',')
    return results

def summarize_documents(text) -> tuple[str, bytes]:
    """
    Summarize the provided documents and return the summary and downloadable file content.
    """
    # Collect context from Streamlit state
    topic = st.session_state.get("detected_topic", "Not specified")
    country = st.session_state.get("selected_country", "Not specified")
    region = st.session_state.get("selected_region", "Not specified")

    # Generate the summary
    response = client.chat.completions.create(
        model='mistralai/Mistral-Small-3.2-24B-Instruct-2506',
        messages=[
            {"role": "system", "content": "You are an analytical research assistant that writes structured, concise summaries."},
            {"role": "user", "content": f"""
            Summarize the following text and make recommendations.

            Context:
            - Topic: {topic}
            - Country: {country}
            - Region: {region}

            Text to summarize:
            {text}
            """}
        ],
        temperature=0.2,
    )

    summary = response.choices[0].message.content.replace('```', '').strip()

    # Create downloadable file
    summary_text = (
        f"Summary Report\n"
        f"===============\n\n"
        f"Topic: {topic}\nCountry: {country}\nRegion: {region}\n\n"
        f"Summary:\n{summary}"
    )
    file_bytes = io.BytesIO(summary_text.encode('utf-8'))

    return summary, file_bytes


def scoring_documents() -> dict:
    '''
    Extracts from the whole index all the documents that contains the codes in the selected codes. 
    This tool extract the Zij scores, converts them into quantiles, and returns them associated with their text.
    '''
    metadata = st.session_state.get(META_ALL_INDEX_KEY)
    selected_codes = st.session_state.get('selected_codes')
    topic = st.session_state.get("detected_topic")
    
    topic = topic.lower()
    selected_meta = []

    if topic == 'technology':
        code_variable = 'CPC_4digit'
        for meta in metadata:
            if meta[code_variable] in selected_codes:
                selected_meta.append(meta)
    
    if topic == 'service':
        code_variable = 'Nice_subclass'
        for meta in metadata:
            if meta[code_variable] in selected_codes:
                selected_meta.append(meta)

    if topic == 'good':
        code_variable = 'Nice_subclass'
        for meta in metadata:
            if meta[code_variable] in selected_codes:
                selected_meta.append(meta)
            
    for meta in selected_meta: # this is needed because of | in the data
        meta['Nice_subclass_keyword'] = meta['Nice_subclass_keyword'].replace('|',',')

    selected_meta_df = pd.DataFrame(selected_meta)
    scores = selected_meta_df['Zij']
    positive_mask = scores > 0
    selected_meta_df = selected_meta_df[positive_mask]
    ranks = rankdata(selected_meta_df['Zij'],method='average')
    quantiles = (ranks - 1) / (sum(positive_mask) - 1)
    selected_meta_df['Quantiles'] = np.round(quantiles,2)
    selected_meta_df = selected_meta_df.sort_values(by='Quantiles',ascending=False)

    if topic == 'technology':
        drop_columns = ['CPC_4digit','CPC_4digit_label_cleaned']
        text_df = selected_meta_df['Nice_subclass_keyword'] + " " + selected_meta_df['Nice_subclass_label_cleaned']
    if topic in ['good','service']:
        drop_columns = ['Nice_subclass','Nice_subclass_keyword','Nice_subclass_label_cleaned']
        text_df = selected_meta_df[['CPC_4digit_label_cleaned']]
    

    results = selected_meta_df.drop(columns=drop_columns)
    #results = results.to_dict(orient='records')
    text_results = text_df.to_dict(orient='records')

    #st.session_state['text_to_summarize']=text_results

    return results,text_results


def selected_codes(selected_codes:list) -> dict:
    '''
    Gets from the prompt the list of codes entered by the user to filter the retrieved documents.'''
    results = st.session_state.get('retrieved documents')
    topic = st.session_state.get("detected_topic")
    
    if topic.lower() == 'technology':
        code_variable = 'CPC_4digit'
    elif topic.lower() == 'service':
        code_variable = 'Nice_subclass'
    elif topic.lower() == 'good':
        code_variable = 'Nice_subclass'

    if selected_codes:
        selected_results = [result for result in results if result[code_variable] in selected_codes]
    else:
        selected_results = results
    st.session_state['selected_results'] = selected_results # we need this only for confirmation purposes.
    st.session_state['selected_codes'] = selected_codes #we need this for the next steps 
    print(selected_codes)
    return {"status":"success",
           "message":("Here are the selected documents: \n"
                      + f"{selected_results}")}

def retrieve_documents(query:str) -> dict:
    '''
    Using the user input, retrieve the relevant documents.
    '''
    topic = st.session_state.get("detected_topic",None)
    selected_region = st.session_state.get("selected_region",None)
    
    if selected_region:
        region_list = st.session_state.get(META_NUTS2_INDEX_KEY)
        for region in region_list:
            if region['NUTS label'] == selected_region:
                region_code = region['NUTS Code']
                break 
        results = retrieve_documents_with_location(region_code,topic,query)
    else:
        results = retrieve_documents_without_location(topic,query)
    
    st.session_state['retrieved documents'] = results

    return {"status": "success", 
            "retrieved_documents": results,
            "next_tool" : "select_documents",
            "message": (
                f" I retrieved the following documents \n"
                + f"{results}"
            ) 
            }  


def display_retrieved_documents():
    docs = st.session_state.get('retrieved documents', [])
    topic = st.session_state.get('detected_topic','technology').lower()

    if not docs:
        st.info("No documents retrieved yet. Click 'Retrieve Documents' first.")
        return

    st.markdown("##### Select documents to keep:")

    selected_codes_list = []

    for idx, doc in enumerate(docs):
        # Determine which field to use for selection
        code_field = 'CPC_4digit' if topic == 'technology' else 'Nice_subclass'
        code = doc[code_field]

        # Unique checkbox key
        checkbox_key = f"doc_checkbox_{idx}_{code}"

        # Display checkbox with document info
        checked = st.checkbox(
            label=f"{code_field}:{code} | "  f"Nice_subclass_keyword:{doc.get('Nice_subclass_keyword','')} | "  f"Nice_subclass_label:{doc.get('Nice_subclass_label_cleaned')} |" f"CPC_4digit_label:{doc.get('CPC_4digit_label_cleaned','')}",
            key=checkbox_key
        )
        if checked:
            selected_codes_list.append(code)

    # Confirm selection button
    if st.button("✅ Confirm Selected Documents"):
        selected_codes(selected_codes_list)
        st.success(f"{len(selected_codes_list)} documents selected for scoring. Continue with scoring!")






