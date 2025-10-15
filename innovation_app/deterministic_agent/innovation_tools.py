import io
import os
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from settings import *
from openai import OpenAI
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer

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

def get_top_lq():
    context = st.session_state.get("detected_context").lower()
    selected_region = st.session_state.get('selected_region')
    region_list = st.session_state.get("META_NUTS2_INDEX_KEY")
    
    if context == 'technology':
        lq_metadata = st.session_state.get("META_TECH_LQ_INDEX_KEY")
        lq_variable = 'tech_lq'
        lq_code_variable = 'cpc'
        label_variable = 'cpc_4digit_label'
    else:
        lq_metadata = st.session_state.get("META_MARKET_LQ_INDEX_KEY")
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        label_variable = 'Nice_subclass_label'
    if selected_region:
        for region in region_list: #finds the NUTS2 code of the region
            if region['NUTS label'] == selected_region:
                region_code = region['NUTS Code']
                break 
        filtered_lq_metadata = pd.DataFrame([meta for meta in lq_metadata if meta['nuts2_code'] == region_code])
        filtered_lq_metadata = filtered_lq_metadata[[lq_code_variable,label_variable,lq_variable]]
        specialization_lq_metadata = filtered_lq_metadata[filtered_lq_metadata[lq_variable]>1]
        specialization_size = specialization_lq_metadata.shape[0]
        if specialization_size > 3:
            st.markdown(f" Here are the top 3 {st.session_state.get('detected_context')} specializations (LQ >1) in {st.session_state.get('selected_region')}:")
            return specialization_lq_metadata.sort_values(lq_variable,ascending=False).head(3)
        elif specialization_size<=3 and specialization_size>0:
            st.markdown(f" There are only {specialization_size} {st.session_state.get('detected_context')} specializations (LQ >1) in {st.session_state.get('selected_region')}:")
            return specialization_lq_metadata.sort_values(lq_variable,ascending=False)
        else:
            st.markdown(f" There are no specializations (LQ >1) in {st.session_state.get('selected_region')}, but the closest (highest LQ) ones are: ")
            return filtered_lq_metadata.sort_values(lq_variable,ascending=False).head(3)
    else:
        lq_metadata = pd.DataFrame(lq_metadata)
        lq_metadata = lq_metadata[['country_en','nuts2',lq_code_variable,label_variable,lq_variable]]
        specialization_lq_metadata = lq_metadata[lq_metadata[lq_variable]>1]
        specialization_size = specialization_lq_metadata.shape[0]
        if specialization_size > 3:
            st.markdown(f"You have not selected a region. Here are the top 3 {st.session_state.get('detected_context')} specializations (LQ >1) in Europe:")
            return specialization_lq_metadata.sort_values(lq_variable,ascending=False).head(3)
        elif specialization_size<=3 and specialization_size>0:
            st.markdown(f"You have not selected a region.. There are only {specialization_size} {st.session_state.get('detected_context')} specializations (LQ >1) in Europe:")
            return specialization_lq_metadata.sort_values(lq_variable,ascending=False)
        else:
            st.markdown(f"You have not selected a region. There are no specializations (LQ >1) in Europe, but the closest (highest LQ) ones are: ")
            return lq_metadata.sort_values(lq_variable,ascending=False).head(3)

def retrieve_documents_with_query(context,query):
    '''
    Using the query, retrieve the relevant documents.
    '''
    #query the doc fais index

    if context.lower() == 'technology':
        faiss_index = st.session_state.get("FAISS_TECH_INDEX_KEY")
        metadata = st.session_state.get("META_TECH_INDEX_KEY")
    elif context.lower() =='service': #bigger than 34
        faiss_index = st.session_state.get("FAISS_SERVICE_INDEX_KEY")
        metadata = st.session_state.get("META_SERVICE_INDEX_KEY")
    elif context.lower() =='good': #less than 34
        faiss_index = st.session_state.get("FAISS_GOOD_INDEX_KEY")
        metadata = st.session_state.get("META_GOOD_INDEX_KEY")

    query_emb = embedding_model.encode([query],convert_to_numpy=True)  # your embedding function
    D, I = faiss_index.search(query_emb, len(metadata))
    results = []
    for j, idx in enumerate(I[0]):  # j iterates 0..k-1
        doc = metadata[idx]
        doc['similarity'] = D[0][j]  # correct alignment
        results.append(doc)

    # FAISS with L2 → smaller distance = better, so we sort ascending
    results = sorted(results, key=lambda x: x['similarity'])
    return results[:5]

def retrieve_documents_with_location_query(context,region_code,query):
    '''
    Using the query and location, retrieve the relevant documents.
    Using locations LQ values filter the relevant documents.
    '''
    #load data based on the context
    if context.lower() == "technology":
        faiss_index = st.session_state.get("FAISS_TECH_INDEX_KEY")
        metadata = st.session_state.get("META_TECH_INDEX_KEY")
        lq_metadata = st.session_state.get("META_TECH_LQ_INDEX_KEY")
        lq_variable = 'tech_lq'
        lq_code_variable = 'cpc'
        code_variable = 'CPC_4digit'
    elif context.lower() =='service':
        faiss_index = st.session_state.get("FAISS_SERVICE_INDEX_KEY")
        metadata = st.session_state.get("META_SERVICE_INDEX_KEY")
        lq_metadata = st.session_state.get("META_MARKET_LQ_INDEX_KEY")
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        code_variable = 'Nice_subclass'
    elif context.lower() =='good':
        faiss_index = st.session_state.get("FAISS_GOOD_INDEX_KEY")
        metadata = st.session_state.get("META_GOOD_INDEX_KEY")
        lq_metadata = st.session_state.get("META_MARKET_LQ_INDEX_KEY")
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        code_variable = 'Nice_subclass'
    else:
        return {"status": "error", "message": f"Unsupported context: {context}"} 
    
    query_emb = embedding_model.encode([query],convert_to_numpy=True)  # your embedding function
    D, I = faiss_index.search(query_emb, len(metadata))
    results = []
    for j, idx in enumerate(I[0]):  # j iterates 0..k-1
        doc = metadata[idx]
        doc['similarity'] = D[0][j]  # correct alignment
        results.append(doc)

    # FAISS with L2 → smaller distance = better, so we sort ascending
    results = sorted(results, key=lambda x: x['similarity'])
    metadata = pd.DataFrame(results)
    print(metadata.head())
    print(metadata.shape)

    #select based on the LQ scores the codes
    lq_results = [meta for meta in lq_metadata if meta.get('nuts2_code') == region_code]
    lq_results = pd.DataFrame(lq_results)
    lq_results = lq_results.drop(columns = ['nuts2_code','country','country_code']).rename(columns={'country_en':'country','nuts2':'region'})
    lq_results = lq_results.merge(right=metadata,how='right',left_on=lq_code_variable,right_on=code_variable)
    lq_results = lq_results.to_dict(orient="records")[:5]
    return lq_results


def retrieve_documents() -> dict:
    '''
    Using the user input, retrieve the relevant documents.
    '''
    context = st.session_state.get("detected_context",None)
    selected_region = st.session_state.get("selected_region",None)
    query = st.session_state.get("query",None)
    
    if selected_region and query:
        region_list = st.session_state.get("META_NUTS2_INDEX_KEY")
        for region in region_list: #finds the NUTS2 code of the region
            if region['NUTS label'] == selected_region:
                region_code = region['NUTS Code']
                break 
        results = retrieve_documents_with_location_query(context,region_code,query)
    else:
        results = retrieve_documents_with_query(context,query)
    
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
    context = st.session_state.get('detected_context','technology').lower()

    if not docs:
        st.info("No documents retrieved yet. Click 'Retrieve Documents' first.")
        return

    st.markdown("##### Select documents to keep:")

    selected_codes_list = []
    for idx, doc in enumerate(docs):
        # Determine which field to use for selection
        code_field = 'CPC_4digit' if context == 'technology' else 'Nice_subclass'
        code = doc[code_field]
        selected_region = st.session_state.get("selected_region",None)

        # Unique checkbox key
        checkbox_key = f"doc_checkbox_{idx}_{code}"

        if context == 'technology':
        # Display checkbox with document info
            if selected_region:
                lq_variable = 'tech_lq'
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n **CPC_4digit_label**: {doc.get('CPC_4digit_label_cleaned','')}  \n  **{lq_variable.capitalize()}**: {np.round(doc.get(lq_variable),2)}  \n  **Distance**: {np.round(doc.get('similarity'),2)}",
                key=checkbox_key
            )
            else:
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n **CPC_4digit_label**: {doc.get('CPC_4digit_label_cleaned')}  \n  **Distance**: {np.round(doc.get('similarity'),2)}",
                key=checkbox_key
            )

        else:
            if selected_region:
                lq_variable = 'market_lq'
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n  **Nice_subclass_keyword**: {doc.get('Nice_subclass_keyword','')}   \n   **Nice_subclass_label**: {doc.get('Nice_subclass_label_cleaned')}  \n  **{lq_variable.capitalize()}**: {np.round(doc.get(lq_variable),2)}  \n  **Distance**: {np.round(doc.get('similarity'),2)}",
                key=checkbox_key
            )
            else:
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n  **Nice_subclass_keyword**: {doc.get('Nice_subclass_keyword','')}   \n   **Nice_subclass_label**: {doc.get('Nice_subclass_label_cleaned')}  \n  **Distance**: {np.round(doc.get('similarity'),2)}",
                key=checkbox_key
            )
        if checked:
            selected_codes_list.append(code)
    # Confirm selection button
    if st.button("✅ Confirm Selected Documents"):
        if selected_codes_list:
            selected_codes(selected_codes_list)
            st.success(f"{len(selected_codes_list)} documents selected for scoring. Continue with scoring!")
        else:
            st.warning("⚠️ Please make at least one selection or restart!")





# TO BE UPDATED
def summarize_documents(text) -> tuple[str, bytes]:
    """
    Summarize the provided documents and return the summary and downloadable file content.
    """
    # Collect context from Streamlit state
    context = st.session_state.get("detected_context", "Not specified")
    country = st.session_state.get("country_code", "Not specified")
    region = st.session_state.get("selected_region", "Not specified")

    user_message = f'''Summarize the following content of type {context} which represents the most 
        relevant documents to users query. It contains the percentiles obtained from the quantiles. 
        Sometines, when user specifies region, they also contain LQ scores which represent the strength of the specialization
        of that region in that field. If LQ score is higher from 1, then that region is specialized in that field
        that follows:\n\n
        This is the context: {context}.
        This is the collection of documents: {text}
        If the type is technology, give your summary from the market perspective (service, good).
        If the type is good or service, then give your summary from the technology perspective.  
        In your repsonse state clearly your perspective.

        A sample response can be of the form:
        From the technology perspective the summary is as follows:
        1. **Speech and Audio Processing**: This includes speech analysis, synthesis, recognition, voice processing, and audio coding and decoding. This is likely to be the most relevant category based on the documents in the 80th percentile or more quantile around 40-50%. 
        Moreover, the average LQ scores for  this topic is significantly higher than 1 meaning the region is higly specialized in this field. 

        2. **Telecommunications**: This includes telephonic communication, transmission of digital information (e.g., telegraphic communication), and wireless communications networks. This category is based on the documents between the 60th and 80th percentiles.
        Even though, this field has high percentile, the avegrage LQ score is less than 1, meaning the region is not specialized in this field. 

        3. **Audio and Acoustic Devices**: This includes loudspeakers, microphones, gramophone pick-ups, deaf-aid sets, and public address systems. This category is based on the documents between the 40th and 60th percentiles.

        4. **Information Storage and Retrieval**: This includes information storage based on relative movement of a record carrier and transduceThis category is based on the documents between the 20th and 40th percentiles.

        5. **Multimedia and Pictorial Communication**: This includes stereophonic systems and pictorial communication (e.g., television). This category is based on the documents in the lowest percentiles.

        '''

    # Generate the summary
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an analytical research assistant that writes structured, concise summaries."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
    )

    summary = response.choices[0].message.content.replace('```', '').strip()

    # Create downloadable file
    summary_text = (
        f"Summary Report\n"
        f"===============\n\n"
        f"context: {context}\nCountry: {country}\nRegion: {region}\n\n"
        f"Summary:\n{summary}"
    )
    file_bytes = io.BytesIO(summary_text.encode('utf-8'))

    return summary, file_bytes


def scoring_documents() -> dict:
    '''
    Extracts from the whole index all the documents that contains the codes in the selected codes. 
    This tool extract the Zij scores, converts them into quantiles, and returns them associated with their text.
    '''
    metadata = st.session_state.get("META_ALL_INDEX_KEY")
    selected_codes = st.session_state.get('selected_codes')
    context = st.session_state.get("detected_context")
    selected_region = st.session_state.get("selected_region",None)
    
    context = context.lower()
    selected_meta = []
    if selected_region:
        if context == 'technology':
            lq_metadata = st.session_state.get("META_MARKET_LQ_INDEX_KEY")
            lq_variable = 'market_lq'
            lq_code_variable = 'Nice_subclass'
            code_variable = 'CPC_4digit'
            code_variable_other = 'Nice_subclass'
        if context == 'service':
            lq_metadata = st.session_state.get("META_TECH_LQ_INDEX_KEY")
            lq_variable = 'tech_lq'
            lq_code_variable = 'cpc'
            code_variable = 'Nice_subclass'
            code_variable_other = 'CPC_4digit'
        if context == 'good':
            lq_metadata = st.session_state.get("META_TECH_LQ_INDEX_KEY")
            lq_variable = 'tech_lq'
            lq_code_variable = 'cpc'
            code_variable = 'Nice_subclass'
            code_variable_other = 'CPC_4digit'

        for meta in metadata:
            if meta[code_variable] in selected_codes:
                selected_meta.append(meta)
        region_list = st.session_state.get("META_NUTS2_INDEX_KEY")
        for region in region_list: #finds the NUTS2 code of the region
            if region['NUTS label'] == selected_region:
                region_code = region['NUTS Code']
                break 
        lq_results = [meta for meta in lq_metadata if meta.get('nuts2_code') == region_code]
        lq_results = pd.DataFrame(lq_results)
        
        selected_meta_df = pd.DataFrame(selected_meta)
        selected_meta_df = lq_results.merge(right=selected_meta_df,how='right',left_on=lq_code_variable,right_on=code_variable_other)
    else:
        if context == 'technology':
            code_variable = 'CPC_4digit'
            for meta in metadata:
                if meta[code_variable] in selected_codes:
                    selected_meta.append(meta)
        
        if context == 'service':
            code_variable = 'Nice_subclass'
            for meta in metadata:
                if meta[code_variable] in selected_codes:
                    selected_meta.append(meta)

        if context == 'good':
            code_variable = 'Nice_subclass'
            for meta in metadata:
                if meta[code_variable] in selected_codes:
                    selected_meta.append(meta)
                
        #for meta in selected_meta: # this is needed because of | in the data
        #    meta['Nice_subclass_keyword'] = meta['Nice_subclass_keyword'].replace('|',',')

        selected_meta_df = pd.DataFrame(selected_meta)

    print(selected_meta_df.head())
    scores = selected_meta_df['Zij']
    positive_mask = scores > 0
    selected_meta_df = selected_meta_df[positive_mask]
    ranks = rankdata(selected_meta_df['Zij'],method='average')
    quantiles = (ranks - 1) / (sum(positive_mask) - 1)
    selected_meta_df['Quantiles'] = np.round(quantiles,2)
    selected_meta_df = selected_meta_df.sort_values(by='Quantiles',ascending=False)

    if context == 'technology':
        if selected_region:
            results = selected_meta_df[['nuts2','country_en','Nice_subclass','Nice_subclass_keyword','Nice_subclass_label_cleaned','Zij',lq_variable,'Quantiles']]
            results = results.rename(columns={'nuts2':'region','country_en':'country','Nice_subclass_label_cleaned':'Nice_subclass_label'})
        else:
            results = selected_meta_df[['Nice_subclass','Nice_subclass_keyword','Nice_subclass_label_cleaned','Zij','Quantiles']]
            results = results.rename(columns={'Nice_subclass_label_cleaned':'Nice_subclass_label'})
    if context in ['good','service']:
        if selected_region:
            results = selected_meta_df[['nuts2','country_en','CPC_4digit','CPC_4digit_label_cleaned','Zij',lq_variable,'Quantiles']]
            results = results.rename(columns={'nuts2':'region','country_en':'country','CPC_4digit_label_cleaned':'CPC_4digit_label'}) 
        else:
            results = selected_meta_df[['nuts2','country_en','CPC_4digit','CPC_4digit_label_cleaned','Zij','Quantiles']]
            results = results.rename(columns={'CPC_4digit_label_cleaned':'CPC_4digit_label'}) 

    return results

def filter_by_percentile_session(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters scored DataFrame based on the percentile set in session state.
    """
    percentile = st.session_state.get('percentile_cutoff', 0.9)
    context = st.session_state.get("detected_context")
    
    if 'Quantiles' not in results_df.columns:
        raise ValueError("DataFrame must have a 'Quantiles' column")
    
    filtered_df = results_df[results_df['Quantiles'] >= percentile]

    if context.lower() == 'technology':
        text_df = pd.DataFrame(filtered_df[['Nice_subclass_keyword','Nice_subclass_label','market_lq','Quantiles']])
    if context.lower() in ['good','service']:
        text_df = filtered_df[['CPC_4digit_label','tech_lq','Quantiles']]
    text_results = text_df.to_dict(orient='records')
    return text_results


def selected_codes(selected_codes:list) -> dict:
    '''
    Gets from the prompt the list of codes entered by the user to filter the retrieved documents.'''
    results = st.session_state.get('retrieved documents')
    context = st.session_state.get("detected_context")
    
    if context.lower() == 'technology':
        code_variable = 'CPC_4digit'
    elif context.lower() == 'service':
        code_variable = 'Nice_subclass'
    elif context.lower() == 'good':
        code_variable = 'Nice_subclass'

    if selected_codes:
        selected_results = [result for result in results if result[code_variable] in selected_codes]
    else:
        selected_results = results
    st.session_state['selected_results'] = selected_results # we need this only for confirmation purposes.
    st.session_state['selected_codes'] = selected_codes #we need this for the next steps 
    return {"status":"success",
           "message":("Here are the selected documents: \n"
                      + f"{selected_results}")}


