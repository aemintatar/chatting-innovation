import io
import re
import json
import pickle
import base64
import requests
import numpy as np
import pandas as pd
import zipfile
from io import BytesIO
import streamlit as st
from settings import *
from openai import OpenAI
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer


# Auxiliary Tools
client = OpenAI(base_url=BASEURL, api_key=APIKEY)

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5",device='cpu')

""" def load_index(index_path: str=None):
    Loads a FAISS index and optional metadata.
    index = None
    r = requests.get(index_path)
    r.raise_for_status()
    if index_path:
        import faiss
            # write to a temporary file (because FAISS needs a real file path)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(r.content)
            tmp.flush()
            index = faiss.read_index(tmp.name)
    
    return index """

import requests
import faiss
import os
import io

def load_index(index_path: str = None):
    """Loads a FAISS index from either a URL or a local file path."""

    if index_path is None:
        return None

    # 🌐 If path is a URL → download
    if str(index_path).startswith("http"):
        r = requests.get(index_path)
        r.raise_for_status()

        # For FAISS binary index read from bytes
        return faiss.read_index_binary(io.BytesIO(r.content))

    # 💾 Else it's a local file → load from disk
    return faiss.read_index(index_path)

""" def load_meta(metadata_path: str = None):
    oads a FAISS index and optional metadata.
    metadata = None
    r = requests.get(metadata_path)
    r.raise_for_status()
    if metadata_path:
        import pickle
        metadata =  pickle.load(BytesIO(r.content))
    return metadata """

def load_meta(metadata_path: str = None):
    """Loads metadata from either a URL or a local file path."""
    
    if metadata_path is None:
        return None
    
    # 🔍 If it's a URL -> download it
    if str(metadata_path).startswith("http"):
        r = requests.get(metadata_path)
        r.raise_for_status()
        return pickle.load(BytesIO(r.content))
    
    # 📦 Otherwise -> read it as a local file
    with open(metadata_path, "rb") as f:
        return pickle.load(f)

def load_country_regions(metadata):
    """Load country-region mappings dynamically."""
    country_regions = {}
    country_names = []
    for region in metadata:
        country = region.get("Country code")
        country_name = region.get("Country name")
        label = region.get("NUTS label")
        if country and country_name and label:
            country_regions.setdefault(country, []).append(label)
            country_names.append(f"{country_name} ({country})")
    return country_regions,sorted(list(set(country_names)))

def get_top_lq():
    context = st.session_state.get("detected_context").lower()
    selected_region = st.session_state.get('selected_region')
    region_list = st.session_state.get("META_NUTS2_INDEX_KEY")
    column_renamig = {'country_en':'Country','nuts2':'Region (NUTS2)','cpc':'Technology code (CPC code 4 digit level)',
                      'cpc_4digit_label':'Technology description','Nice_subclass':'Product code','Nice_subclass_label':'Product description'}
    column_droping = ['tech_lq','market_lq']
    
    if context == 'technology':
        lq_metadata = st.session_state.get("META_TECH_LQ_INDEX_KEY")
        lq_variable = 'tech_lq'
        lq_code_variable = 'cpc'
        label_variable = 'cpc_4digit_label' #from lq dataset, not clean
    else:
        lq_metadata = st.session_state.get("META_MARKET_LQ_INDEX_KEY")
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        label_variable = 'Nice_subclass_label' #from lq dataset, not clean
    if selected_region:
        for region in region_list: #finds the NUTS2 code of the region
            if region['NUTS label'] == selected_region:
                region_code = region['NUTS Code']
                break 
        filtered_lq_metadata = pd.DataFrame([meta for meta in lq_metadata if meta['nuts2_code'] == region_code])
        if filtered_lq_metadata.empty:
             st.markdown(f" No {st.session_state.get('detected_context')} specializations found in {st.session_state.get('selected_region')}. Please select another set of parameters.")
        else:
            filtered_lq_metadata = filtered_lq_metadata[[lq_code_variable,label_variable,lq_variable]]
            specialization_lq_metadata = filtered_lq_metadata[filtered_lq_metadata[lq_variable]>1]
            specialization_size = specialization_lq_metadata.shape[0]
            if specialization_size > 3:
                st.markdown(f" Based on the parameters, here are the top 3 {st.session_state.get('detected_context')} specializations in {st.session_state.get('selected_region')}:")
                specialization_lq_metadata = specialization_lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop= True).head(3)
                specialization_lq_metadata.index = [1,2,3]
                specialization_lq_metadata = specialization_lq_metadata.rename(columns=column_renamig)
                specialization_lq_metadata = specialization_lq_metadata.drop(columns=column_droping,errors='ignore')
                return specialization_lq_metadata
            elif specialization_size<=3 and specialization_size>0:
                st.markdown(f" There are only {specialization_size} {st.session_state.get('detected_context')} specializations in {st.session_state.get('selected_region')}:")
                specialization_lq_metadata = specialization_lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop=True).head(3)
                specialization_lq_metadata.index = range(1,len(specialization_lq_metadata))
                specialization_lq_metadata = specialization_lq_metadata.rename(columns=column_renamig)
                specialization_lq_metadata = specialization_lq_metadata.drop(columns=column_droping,errors='ignore')
                return specialization_lq_metadata
            else:
                st.markdown(f" There are no specializations in {st.session_state.get('selected_region')}, but the closest ones are: ")
                filtered_lq_metadata = filtered_lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop=True).head(3)
                filtered_lq_metadata.index = [1,2,3]
                filtered_lq_metadata = filtered_lq_metadata.rename(columns=column_renamig)
                filtered_lq_metadata = filtered_lq_metadata.drop(columns=column_droping,errors='ignore')
                return filtered_lq_metadata
    #else:
        # Plan change: We do nbot show the any list if no country and region is selected.
        # activate if the plans change.
        #lq_metadata = pd.DataFrame(lq_metadata)
        #lq_metadata = lq_metadata[['country_en','nuts2',lq_code_variable,label_variable,lq_variable]]
        #specialization_lq_metadata = lq_metadata[lq_metadata[lq_variable]>1]
        #specialization_size = specialization_lq_metadata.shape[0]
        #if specialization_size > 3:
        #    st.markdown(f"You have not selected a region. Here are the top 3 {st.session_state.get('detected_context')} specializations in Europe:")
        #    specialization_lq_metadata = specialization_lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop=True).head(3)
        #    specialization_lq_metadata.index = [1,2,3]
        #    specialization_lq_metadata = specialization_lq_metadata.rename(columns=column_renamig)
        #    specialization_lq_metadata = specialization_lq_metadata.drop(columns=column_droping,errors='ignore')
        #    return specialization_lq_metadata
        #elif specialization_size<=3 and specialization_size>0:
        #    st.markdown(f"You have not selected a region. There are only {specialization_size} {st.session_state.get('detected_context')} specializations in Europe:")
        #    specialization_lq_metadata = specialization_lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop=True)
        #    specialization_lq_metadata.index = range(1,len(specialization_lq_metadata))
        #    specialization_lq_metadata = specialization_lq_metadata.rename(columns=column_renamig)
        #    specialization_lq_metadata = specialization_lq_metadata.drop(columns=column_droping,errors='ignore')
        #    return specialization_lq_metadata
        #else:
        #    st.markdown(f"You have not selected a region. There are no specializations in Europe, but the closest ones are: ")
        #    lq_metadata = lq_metadata.sort_values(lq_variable,ascending=False).reset_index(drop=True).head(3)
        #    lq_metadata.index = [1,2,3]
        #    filtered_lq_metadata = filtered_lq_metadata.rename(columns=column_renamig)
        #    filtered_lq_metadata = filtered_lq_metadata.drop(columns=column_droping,errors='ignore')
        #    return lq_metadata

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

    query_emb = embedding_model.encode([query],convert_to_numpy=True)
    D, I = faiss_index.search(query_emb, len(metadata))
    results = []
    for j, idx in enumerate(I[0]):  
        doc = metadata[idx]
        doc['similarity'] = D[0][j] 
        results.append(doc)

    # FAISS with L2 → smaller distance is better, so sort ascending
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
    
    query_emb = embedding_model.encode([query],convert_to_numpy=True) 
    D, I = faiss_index.search(query_emb, len(metadata))
    results = []
    for j, idx in enumerate(I[0]):  
        doc = metadata[idx]
        doc['similarity'] = D[0][j]  
        results.append(doc)

    # FAISS with L2 → smaller distance is better, so sort ascending
    results = sorted(results, key=lambda x: x['similarity'])
    metadata = pd.DataFrame(results)

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
    context = st.session_state.get('detected_context','technology').lower() #LOOK: why 'technology' is fallback here?

    if not docs:
        st.info("No documents retrieved yet. Click 'Retrieve Documents' first.")
        return

    st.markdown("##### Select documents to keep:")
    st.write("Results are displayed starting with the most relevant matches to your query." )

    selected_codes_list = []
    for idx, doc in enumerate(docs):
        # Determine which field to use for selection
        code_field = 'CPC_4digit' if context == 'technology' else 'Nice_subclass'
        code = doc[code_field]
        if code_field == 'CPC_4digit':
            code_field = 'Technology code'
        else:
            code_field = 'Product code'
        selected_region = st.session_state.get("selected_region",None)

        # Unique checkbox key
        checkbox_key = f"doc_checkbox_{idx}_{code}"

        if context == 'technology':
        # Display checkbox with document info
            if selected_region:
                lq_variable = 'tech_lq'
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n **Technology description**: {doc.get('CPC_4digit_label_cleaned','')} ",
                key=checkbox_key
            )
            else:
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n **Technology description**: {doc.get('CPC_4digit_label_cleaned','')}",
                key=checkbox_key
            )

        else:
            if selected_region:
                lq_variable = 'market_lq'
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n  **Product description (keywords)**: {doc.get('Nice_subclass_keyword','')}   \n   **Product description**: {doc.get('Nice_subclass_label_cleaned')}",
                key=checkbox_key
            )
            else:
                checked = st.checkbox(
                label=f"**{code_field}**: {code}  \n  **Product Description (keywords)**: {doc.get('Nice_subclass_keyword','')}   \n   **Product description**: {doc.get('Nice_subclass_label_cleaned')}",
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

    scores = selected_meta_df['Zij']
    positive_mask = scores > 0
    selected_meta_df = selected_meta_df[positive_mask]
    ranks = rankdata(selected_meta_df['Zij'],method='average')
    quantiles = (ranks - 1) / (sum(positive_mask) - 1)
    selected_meta_df['Quantiles'] = np.round(quantiles,2)
    selected_meta_df = selected_meta_df.sort_values(by='Quantiles',ascending=False)

    if context == 'technology':
        if selected_region:
            results = selected_meta_df[['Nice_subclass','Nice_subclass_keyword','Nice_subclass_label_cleaned','Zij',lq_variable,'Quantiles']]
            results = results.rename(columns={'Nice_subclass_label_cleaned':'Nice_subclass_label'})
        else:
            results = selected_meta_df[['Nice_subclass','Nice_subclass_keyword','Nice_subclass_label_cleaned','Zij','Quantiles']]
            results = results.rename(columns={'Nice_subclass_label_cleaned':'Nice_subclass_label'})
    if context in ['good','service']:
        if selected_region:
            results = selected_meta_df[['CPC_4digit','CPC_4digit_label_cleaned','Zij',lq_variable,'Quantiles']]
            results = results.rename(columns={'CPC_4digit_label_cleaned':'CPC_4digit_label'}) 
        else:
            results = selected_meta_df[['CPC_4digit','CPC_4digit_label_cleaned','Zij','Quantiles']]
            results = results.rename(columns={'CPC_4digit_label_cleaned':'CPC_4digit_label'}) 

    return results

def filter_by_quantile_session(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Keep only highly relevant documents (based on quantile).
    Step 2: Identify technologies where the selected region lacks specialization (LQ < 1).
    These weak areas will later be used to find benchmark regions.
    """
    quantile_cutoff = st.session_state.get('quantile_cutoff', 0.9)
    context = st.session_state.get("detected_context", "Not specified")
    selected_region = st.session_state.get("selected_region", None)
    if 'Quantiles' not in results_df.columns:
        raise ValueError("DataFrame must have a 'Quantiles' column")
    
    #Step1
    filtered_df = results_df[results_df['Quantiles'] >= quantile_cutoff]
    
    #Step2
    if selected_region:
        if context == 'technology':
            lq_variable = 'market_lq'
        else:
            lq_variable = 'tech_lq'
    st.session_state['filtered_docs'] = filtered_df
    return filtered_df

def display_filtered_documents(filtered_docs):

    if filtered_docs.empty:
        st.info("No documents to display.")
    else:
        # ---- Select documents ----
        df = filtered_docs.copy()

        # Add checkbox column if it doesn't exist
        if "Select" not in df.columns:
            df.insert(0, "Select", False)

        # Editable table with checkboxes
        edited_df = st.data_editor(
            df,
            hide_index=True,
            width=1600,
            key="document_selector",
            column_config= {
                "Select":st.column_config.CheckboxColumn(width=10),
                'Nice_subclass':st.column_config.TextColumn(label="Product code",width=20),
                'Nice_subclass_keyword':st.column_config.TextColumn(label="Product description (keywords)",width=300),
                'Nice_subclass_label':st.column_config.TextColumn(label="Product description",width=600),
                'CPC_4digit':st.column_config.TextColumn(label="Technology code",width=20),
                'CPC_4digit_label':st.column_config.TextColumn(label="Technology description",width=1200),
            },
            column_order=["Select","Nice_subclass","Nice_subclass_keyword","Nice_subclass_label","CPC_4digit","CPC_4digit_label"]
        )

        selected_docs = edited_df[edited_df["Select"]]
        #st.dataframe(selected_docs)  # Display selected documents
        if st.button("✅ Confirm selected documents"):
            if len(selected_docs) > 5:
                st.warning("⚠️ You can select a maximum of 5 documents.")
            elif len(selected_docs) == 0:
                st.warning("⚠️ Please select at least one document to proceed.")
            else:
                selected_docs = edited_df[edited_df["Select"]]
                st.session_state["selected_docs"] = selected_docs
                st.success(f"{len(selected_docs)} documents have been selected. Continue with summarization!")


def specialized_regions():  #per_document_version
    """
    Identify regions that specialize (LQ ≥ 1) in technologies where the selected
    region is not specialized (LQ < 1).

    Outputs:
    - general_specialized: globally strongest regions
    - local_specialized: geographically closest specialized regions
    """
    # Collect context from Streamlit state
    context = st.session_state.get("detected_context", "Not specified")
    selected_region = st.session_state.get("selected_region", "Not specified")
    region_list = pd.DataFrame(st.session_state.get("META_NUTS2_INDEX_KEY"))
    selected_code = region_list['NUTS Code'][region_list['NUTS label']==selected_region].values[0] #finds the NUTS2 code of the region


    distance_df = pd.DataFrame(st.session_state['META_DISTANCE_INDEX_KEY'])
    distance_df = distance_df[distance_df['nuts2_1'] == selected_code]
    if context == 'technology':
        lq_variable = 'market_lq'
        lq_code_variable = 'Nice_subclass'
        code_variable = 'Nice_subclass'
        lq_metadata = pd.DataFrame(st.session_state.get("META_MARKET_LQ_INDEX_KEY"))
    else:
        lq_variable = 'tech_lq'
        lq_code_variable = 'cpc'
        code_variable = 'CPC_4digit'
        lq_metadata = pd.DataFrame(st.session_state.get("META_TECH_LQ_INDEX_KEY"))
    
    filtered_df = st.session_state.get("selected_docs",pd.DataFrame())
    filtered_codes = filtered_df[code_variable].tolist()
    print(f'Filtered documents:\n{filtered_codes}')

    weak_docs = filtered_df[filtered_df[lq_variable] < 0.7]
    weak_codes = weak_docs[code_variable].tolist()
    print(f'Weak codes:\n{weak_codes}')

    target_codes = list(set(weak_codes + filtered_codes))
    print(f'Target codes for specialization search:\n{target_codes}')
   
    #highest LQ and highest LQ with shortest distance
    candidate_regions = lq_metadata[lq_metadata[lq_code_variable].isin(target_codes)]
    
     # ---- Keep only specialized regions ----
    specialized_regions = candidate_regions[candidate_regions[lq_variable] >= 0.7]

    
    global_specialized = specialized_regions.groupby(lq_code_variable, group_keys=False).apply(lambda g: g.nlargest(3, lq_variable))
    
    # Keep only technologies appearing in filtered documents
    global_specialized = global_specialized[
        global_specialized[lq_code_variable].isin(filtered_df[code_variable])
    ]


    specialized_with_distance = specialized_regions.merge(
        distance_df,
        left_on="nuts2_code",
        right_on="nuts2_2"
    )
    
    
    specialized_with_distance = specialized_with_distance[
        [lq_code_variable, "nuts2_2", "nuts2", "country_en", lq_variable, "distance_km"]
    ]


    merged_docs = filtered_df.merge(
        specialized_with_distance,
        left_on=code_variable,
        right_on=lq_code_variable,
        suffixes=["_origin", "_closest"]
    )

    closest_specialized = (
        merged_docs
        .groupby(code_variable, group_keys=False)
        .apply(lambda g: g.nsmallest(3, "distance_km"))
    )

    closest_specialized = closest_specialized[
        [code_variable, "nuts2_2", "nuts2", "country_en", lq_variable+'_closest', "distance_km"]
    ]

    # ---- Save results ----
    st.session_state["general_specialized"] = global_specialized
    st.session_state["local_specialized"] = closest_specialized
    return global_specialized, closest_specialized

def without_specialized_regions():
    """This will be used when there is no LQ score lower than 1, and thus no specialized regions to recommend."""
    filtered_df = st.session_state.get("selected_docs",pd.DataFrame())
    st.session_state['general_specialized'] = None
    st.session_state['local_specialized'] = None
    return None, None



def summarize_documents() -> tuple[str, bytes]: #per_document_version
    """
    Summarize the provided documents and return the summary and downloadable file content.
    """
    # Collect context from Streamlit state
    context = st.session_state.get("detected_context", "Not specified")
    country = st.session_state.get("country_code", "Not specified")
    region = st.session_state.get("selected_region", "Not specified")
    selected_df = st.session_state.get("selected_docs")
    general_specialized_df = st.session_state.get("general_specialized")
    local_specialized_df = st.session_state.get("local_specialized")
    general_specialized_documents = general_specialized_df.to_dict(orient='records')
    local_specialized_documents = local_specialized_df.to_dict(orient='records')

    summary = {}
    print('Selected documents for summarization:')
    print(selected_df.head())
    print('General specialized regions:')
    print(general_specialized_df.head())
    print('Local specialized regions:')
    print(local_specialized_df.head())
    #Complete this for loop.
    for i,r in selected_df.iterrows():
        temp = {}
        #create the user text
        if context.lower() == 'technology':
            lq_variable = 'market_lq'
            lq_code_variable = 'Nice_subclass'
            code_variable = 'Nice_subclass'
            code = r[lq_code_variable]
            if region:
                text_df = r[['Nice_subclass_keyword','Nice_subclass_label','market_lq','Quantiles']]
                general_specialized_documents = general_specialized_df[general_specialized_df[lq_code_variable] == code].to_dict(orient='records')
                local_specialized_documents = local_specialized_df[local_specialized_df[code_variable] == code].to_dict(orient='records') 
            else:
                text_df = r[['Nice_subclass_keyword','Nice_subclass_label','Quantiles']]
                general_specialized_documents = None
                local_specialized_documents = None
        if context.lower() in ['good','service']:
            lq_variable = 'tech_lq'
            lq_code_variable = 'cpc'
            code_variable = 'CPC_4digit'
            code = r[code_variable]
            if region:
                text_df = r[['CPC_4digit_label','tech_lq','Quantiles']]
                general_specialized_documents = general_specialized_df[general_specialized_df[lq_code_variable] == code].to_dict(orient='records')
                local_specialized_documents = local_specialized_df[local_specialized_df[code_variable] == code].to_dict(orient='records')   
            else:
                text_df = r[['CPC_4digit_label','Quantiles']]
                general_specialized_documents = None
                local_specialized_documents = None


        text = text_df.to_dict()

        user_message = f'''Summarize the following content which represents the most 
            relevant documents to users query and auxiliary documents related to the top locations and closest top locations when location information is present. 
            They contain the quantiles obtained from the scores representing the relationships between CPC codes and Nice codes, LQ scores representing the strength 
            of the region's specialization in that field. If LQ score is higher from 1, then that region is specialized in that field.
            When LQ scores are lower than 1 for some codes, you are expected to ALWAYS recommend top 3 specializations using the general specialized documents 
            and also recommend closet top 3 specializations using the local specialized documents.
            When you refer to those top 3 locations do not refer to them using their NUTS2 code or country names. Use ONLY their region/nuts2 names as known in public. 
            Include distances in KM to your response to be transparent.
            Do not include in the summary the quantiles, but only the relative position of the documents (e.g. top 1, top 2, top 3, etc.).
            Do not include in the summary the LQ scores, but only if the region is specialized or not, and if not, recommend the closest specialized regions.
            \n

            If the context is technology, give your summary from the market perspective (service, good).
            If the context is good or service, then give your summary from the technology perspective. 
            However, in your repsonse do not state your perspective. 
            
            Learn from the samples below, how to respond and organize the response:

            In case LQ scores are presents, a sample response can be of the form, assuming context is service or good

            **Rental and Hire Services: Construction Equipment, Cleaning Machines, Industrial Apparatus** 
            - The region is **not specialized** in this field.
            - In Europe, the top 3 locations specialized in this field are
                - Île de France (France)
            - The loacation above is also the closest in this filed with a distance of 1050.04 km to Burgenland.

            **Power-Operated Machines and Appliances: Food Processing, Kitchen Tasks, Industrial Applications**
            - The region is **specialized** in this field.
            - In Europe, the top 3 locations specialized in this field are:
                - Cataluña (Spain)
                - Toscana (Italy)
                - Freiburg (Germany)
            - The closest top 3 specialized locations to the region are:
                - Freiburg (Germany) with a distance of 1494.17 km,
                - Piemonte (Italy) with a distance of 1765.02 km,
                - Toscana (Italy) with a distance of 1781.14 km.

            **Pumps, Compressors, Blowers, Air Handling Equipment: Industrial and Mechanical Applications**
            - The region is **not specialized** in this field. 
            - In Europe, the top 3 locations specialized in this field are:
                - Stuttgart (Germany), 
                - Emilia-Romagna (Italy), 
                - Düsseldorf (Germany). 
            - The closest top 3 specialized locations to Burgenland (Austria) are:
                - Veneto (Italy) with a distance of 415.52 km, 
                - Stuttgart (Germany) with a distance of 537.03 km,
                - Emilia-Romagna (Italy) with a distance of 540.74 km.
            
            In case LQ scores are missing, a sample response can be of the form, assuming context is service or good:
            
            **Rental and Hire Services: Construction Equipment, Cleaning Machines, Industrial Apparatus**
            - It cannot be determined whether the region is specialized in this category.
            - In Europe, the top 3 locations specialized in this field are
                - Île de France (France)
            - The loaction above is also the closest in this filed with a distance of 105

            **Power-Operated Machines and Appliances: Food Processing, Kitchen Tasks, Industrial Applications**
            - It cannot be determined whether the region is specialized in this category.
            - In Europe, the top 3 locations specialized in this field are:
                - Stuttgart (Germany)
                - Emilia-Romagna (Italy)
                - Düsseldorf (Germany)
            - The closest top 3 specialized locations to Burgenland (Austria) are:
                - Veneto (Italy) with a distance of 415.52 km, 
                - Stuttgart (Germany) with a distance of 537.03 km,     
                - Emilia-Romagna (Italy) with a distance of 540.74 km.

            Return your response in TWO parts:

            1. A human-readable summary as in the samples above

            2. A JSON object with the following structure:
                title:
                is_specialized: true/false,
                top_regions: [top_region1, top_region2, top_region3],
                closest_regions: [closestregion1, closest_region2, closest_region3]
            


            Return ONLY valid JSON for part 2. Do not include explanations inside the JSON.
            Split the two responses with a clear separator using "### JSON ###" in between.
            Here are the documents needed for the summary:
            Context: {context}
            Collection of documents: {text}
            General specialized documents: {general_specialized_documents}
            Local specialized documents: {local_specialized_documents}
            '''
            

        # Generate the summary
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an analytical research assistant that writes structured, concise summaries."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
        )

        clean_response = response.choices[0].message.content.replace('```', '').strip()
        temp['text'] = clean_response.split("### JSON ###")[0].strip()
        temp['local_code'] = [doc['nuts2_2'] for doc in local_specialized_documents]
        temp['global_code'] = [doc['nuts2_code'] for doc in general_specialized_documents]
        summary.update({code: temp})
    st.session_state['summary'] = summary
    text = f"From the {context} perspective the summary is as follows:\n\n"
    for code, content in summary.items():
        text += content['text'] + '\n\n'
    summary_text = (
        f"**Context:** {context.capitalize()}\n\n**Country:** {country}\n\n**Region:** {region}\n\n"
        f"**Summary:**\n\n{text}"
    )
    st.session_state['summary_text'] = summary_text
    return summary



def summary_download():
    summary_text = st.session_state.get("summary_text")
    # Create downloadable file
    file_bytes = io.BytesIO(summary_text.encode('utf-8'))
    return file_bytes


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

