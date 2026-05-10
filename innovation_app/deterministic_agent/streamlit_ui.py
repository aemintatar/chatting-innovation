import os
import gdown
import streamlit as st
import geopandas as gpd
import pydeck as pdk
from settings import *
from datetime import datetime
from innovation_tools import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch

from huggingface_hub import hf_hub_download


# =========================
# 🔧 PAGE CONFIG
# =========================
st.set_page_config(page_title="Chatting Innovation\\Bridging Invention and Market Innovation", layout="wide") # Configures the browser tab title and page layout.
st.title("Chatting Innovation") # Main title of the app.
st.subheader("Bridging Invention and Market Innovation")
st.markdown("""
            The PAT2TM chatbot, powered by Mistral and Streamlit, builds on the work of
Abbasiharofteh, Castaldi, and Petralia (2025 &amp; forthcoming), which establishes a
comprehensive concordance between patent and trademark classes.
            
The PAT2TM chatbot offers data-driven insights for practitioners, startups, and
policymakers seeking to bridge inventions and market opportunities. By mapping
technologies to goods and services, it helps identify diversification paths, niche
market potentials, and strategic partnerships.
            import plotly.graph_objects as go
You can explore the connections between technologies and markets in two ways:
* Enter a technology to discover the goods and services it enables.
* Enter a good or service to identify the technologies required for its development and market application.

The chatbot returns a summary of the relevant goods, services, or technologies,
along with the strength of their associations derived from the patent-to-
trademark concordance.
            
💡 **Tip:** More specific queries yield better results. For example, instead of typing
“Drones”, try “Drone Power System” or “Drone Flight Controller” for more accurate
and meaningful associations.
If you use the PAT2TM chatbot or related data, please cite:
* Abbasiharofteh, Milad; Castaldi, Carolina; Petralia, Sergio (2025). From
technologies to markets: A concordance between patent and trademark
classes. , https://doi.org/10.7910/DVN/JD7JIL, Harvard Dataverse, V1
* Abbasiharofteh, Milad; Castaldi, Carolina; Petralia, Sergio (forthcoming). From
technologies to markets: A concordance between patent and trademark
classes. Scientific Data.
* Abbasiharofteh, Milad; Tatar, Emin (forthcoming). Chatting Innovation:
Bridging Invention and Market Innovation. arXiv preprint arXiv:xxx.xxx.""")
                    
from huggingface_hub import hf_hub_download
import streamlit as st

@st.cache_resource
def load_all_data_from_drive():
    st.write("📦 Loading document indexes...")

    REPO_ID = "atatar/innovation_data"

    HF_FILES = {
        "META_ALL_INDEX_PATH": "allmetadata.pkl",  
        "FAISS_TECH_INDEX_PATH": "faiss_tech_index.bin",
        "META_TECH_INDEX_PATH": "techmetadata.pkl",
        "FAISS_SERVICE_INDEX_PATH": "faiss_service_index.bin",
        "META_SERVICE_INDEX_PATH": "servicemetadata.pkl",
        "FAISS_GOOD_INDEX_PATH": "faiss_good_index.bin",
        "META_GOOD_INDEX_PATH": "goodmetadata.pkl",
        "META_MARKET_LQ_INDEX_PATH": "market_lq_metadata.pkl",
        "META_TECH_LQ_INDEX_PATH": "tech_lq_metadata.pkl",
        "META_DISTANCE_INDEX_PATH": "distance.pkl",
        "META_NUTS2_INDEX_PATH": "nuts2.pkl",
        "RG_SHAPEFILE_NUTS2_PATH": "NUTS_RG_03M_2013.shp",
        "RG_SHAPEFILE_AUX1_PATH": "NUTS_RG_03M_2013.sbn",
        "RG_SHAPEFILE_AUX2_PATH": "NUTS_RG_03M_2013.sbx",
        "RG_SHAPEFILE_AUX3_PATH": "NUTS_RG_03M_2013.shp.xml",
        "RG_SHAPEFILE_AUX4_PATH": "NUTS_RG_03M_2013.shx",
        "RG_SHAPEFILE_AUX5_PATH": "NUTS_RG_03M_2013.dbf",
        "RG_SHAPEFILE_AUX6_PATH": "NUTS_RG_03M_2013.prj",
        "LB_SHAPEFILE_NUTS2_PATH": "NUTS_LB_03M_2013.shp",
        "LB_SHAPEFILE_AUX1_PATH": "NUTS_LB_03M_2013.sbn",
        "LB_SHAPEFILE_AUX2_PATH": "NUTS_LB_03M_2013.sbx",
        "LB_SHAPEFILE_AUX3_PATH": "NUTS_LB_03M_2013.shp.xml",
        "LB_SHAPEFILE_AUX4_PATH": "NUTS_LB_03M_2013.shx",
        "LB_SHAPEFILE_AUX5_PATH": "NUTS_LB_03M_2013.dbf",
        "LB_SHAPEFILE_AUX6_PATH": "NUTS_LB_03M_2013.prj",
        "LB_SHAPEFILE_AUX7_PATH": "NUTS_LB_03M_2013.cpg",
    }

    def local_file(filename):
        return hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",     # <-- REQUIRED
            force_download=False,
            local_files_only=False
        )

    st.write("⚙️ Reading data files into memory...")

    all_meta = load_meta(local_file(HF_FILES["META_ALL_INDEX_PATH"]))
    tech_index = load_index(local_file(HF_FILES["FAISS_TECH_INDEX_PATH"]))
    tech_meta = load_meta(local_file(HF_FILES["META_TECH_INDEX_PATH"]))
    service_index = load_index(local_file(HF_FILES["FAISS_SERVICE_INDEX_PATH"]))
    service_meta = load_meta(local_file(HF_FILES["META_SERVICE_INDEX_PATH"]))
    good_index = load_index(local_file(HF_FILES["FAISS_GOOD_INDEX_PATH"]))
    good_meta = load_meta(local_file(HF_FILES["META_GOOD_INDEX_PATH"]))
    market_lq_meta = load_meta(local_file(HF_FILES["META_MARKET_LQ_INDEX_PATH"]))
    tech_lq_meta = load_meta(local_file(HF_FILES["META_TECH_LQ_INDEX_PATH"]))
    distance_index = load_meta(local_file(HF_FILES["META_DISTANCE_INDEX_PATH"]))
    nuts2_meta = load_meta(local_file(HF_FILES["META_NUTS2_INDEX_PATH"]))
    rg_shapefile_aux1 = local_file(HF_FILES["RG_SHAPEFILE_AUX1_PATH"])
    rg_shapefile_aux2 = local_file(HF_FILES["RG_SHAPEFILE_AUX2_PATH"])
    rg_shapefile_aux3 = local_file(HF_FILES["RG_SHAPEFILE_AUX3_PATH"])
    rg_shapefile_aux4 = local_file(HF_FILES["RG_SHAPEFILE_AUX4_PATH"])
    rg_shapefile_aux5 = local_file(HF_FILES["RG_SHAPEFILE_AUX5_PATH"])
    rg_shapefile_aux6 = local_file(HF_FILES["RG_SHAPEFILE_AUX6_PATH"])
    rg_shapefile_nuts2 = gpd.read_file(local_file(HF_FILES["RG_SHAPEFILE_NUTS2_PATH"]))
    lb_shapefile_aux1 = local_file(HF_FILES["LB_SHAPEFILE_AUX1_PATH"])
    lb_shapefile_aux2 = local_file(HF_FILES["LB_SHAPEFILE_AUX2_PATH"])
    lb_shapefile_aux3 = local_file(HF_FILES["LB_SHAPEFILE_AUX3_PATH"])
    lb_shapefile_aux4 = local_file(HF_FILES["LB_SHAPEFILE_AUX4_PATH"])
    lb_shapefile_aux5 = local_file(HF_FILES["LB_SHAPEFILE_AUX5_PATH"])
    lb_shapefile_aux6 = local_file(HF_FILES["LB_SHAPEFILE_AUX6_PATH"])
    lb_shapefile_aux7 = local_file(HF_FILES["LB_SHAPEFILE_AUX7_PATH"])
    lb_shapefile_nuts2 = gpd.read_file(local_file(HF_FILES["LB_SHAPEFILE_NUTS2_PATH"]))

    return {
        "META_ALL_INDEX_KEY": all_meta,
        "FAISS_TECH_INDEX_KEY": tech_index,
        "META_TECH_INDEX_KEY": tech_meta,
        "FAISS_SERVICE_INDEX_KEY": service_index,
        "META_SERVICE_INDEX_KEY": service_meta,
        "FAISS_GOOD_INDEX_KEY": good_index,
        "META_GOOD_INDEX_KEY": good_meta,
        "META_MARKET_LQ_INDEX_KEY": market_lq_meta,
        "META_TECH_LQ_INDEX_KEY": tech_lq_meta,
        "META_DISTANCE_INDEX_KEY": distance_index,
        "META_NUTS2_INDEX_KEY": nuts2_meta,
        "RG_SHAPEFILE_NUTS2_KEY": rg_shapefile_nuts2,
        "LB_SHAPEFILE_NUTS2_KEY": lb_shapefile_nuts2
    }


# 🧠 Initialize data once
if "data_loaded" not in st.session_state:
    with st.spinner("Loading all data from Hugging Face..."):
        data_dict = load_all_data_from_drive()
        for k, v in data_dict.items():
            st.session_state[k] = v
        st.session_state["data_loaded"] = True
        st.success("✅ Data loaded successfully!")

st.divider() # A visual separator.
st.markdown("#### Search Parameters")
#  context / country / region selectors (main frame)

col1, col2, col3 = st.columns(3)

with col1:
    context_value = st.selectbox(
    "**Context** (required)",["-- None --", "Technology", "Service", "Good"],
    index=0,
    key=f"sidebar_context"
)
with col2:
    nuts2_meta = st.session_state.get("META_NUTS2_INDEX_KEY")
    country_regions = load_country_regions(nuts2_meta)[0]
    country_names = load_country_regions(nuts2_meta)[1]
    country_options = ["-- None --"] + country_names
    country_value = st.selectbox("**Country** (optional)", 
                           country_options, 
                           index=0,
                           key=f"sidebar_country")
with col3:
    region_options = ["-- None --"]
    if country_value and not country_value.startswith("--"):
        country_code = country_value.split("(")[-1].split(")")[0].strip()
        region_options += sorted(country_regions.get(country_code, []))
    region_options = [option for option in region_options if option!='Extra-Regio NUTS 2']
    region_value = st.selectbox("**Region** (optional)", 
                          region_options, 
                          index=0,
                          key=f"sidebar_region")
# Apply context button
if st.button("Apply Parameters"):
    st.session_state["detected_context"] = None if context_value.startswith("--") else context_value.lower()
    st.session_state["country_code"] = None if country_value.startswith("--") else country_value
    st.session_state["selected_region"] = None if region_value.startswith("--") else region_value
    st.success("Parameters applied!")

if 'detected_context' in st.session_state and st.session_state.get('country_code') and st.session_state.get('selected_region'):
    st.write('#### Specializations')
    df = get_top_lq()
    st.session_state['specialization'] = df
    st.dataframe(st.session_state.get('specialization'))
    

st.divider() # A visual separator.
st.markdown("#### Query")
prompt = st.text_area("Enter your product or technology idea.",key="query")


# =========================
# 🔹 STEP 1: Interpret and retrieve 
# =========================
if st.button("🔍 Retrieve Documents"):
    if not st.session_state.get("detected_context"):
        st.warning("⚠️ Please select a context first.")
    elif st.session_state.get("country_code") and not st.session_state.get("selected_region"):
        st.warning("⚠️ Please select a region as well!")
    else:
        query = st.session_state.get("query")
        if not query.strip():
            st.warning("⚠️ Please enter an idea before retrieving documents.")
        else:
            retrieve_documents()
            docs = st.session_state.get('retrieved documents', [])
            st.success(f"✅ {len(docs)} documents retrieved.")


# =========================
# 🔹 STEP 2: Display & Select
# =========================
if st.session_state.get('retrieved documents'):
    display_retrieved_documents()


# =========================
# 🔹 STEP 3: Scoring
# =========================
if st.session_state.get('selected_codes'):
    if st.button("📊 Score Selected Documents"):
        scored_docs = scoring_documents()  # scoring_tool saves results to session_state['text_to_summarize']
        st.success("✅ Documents scored successfully!")
        st.session_state['scored_docs'] = scored_docs
    if 'scored_docs' in st.session_state:
        st.write("#### Top Matches")
        if st.session_state.get('selected_region'):
            st.write(f'Based on the PAT2TM concordance, the top matches are shown for {st.session_state.get("selected_region")}, {st.session_state.get("country_code")}, with relevance decreasing further down.')
            st.write('You can select up to five documents for summarization.')
        else:
            st.write(f'Based on the PAT2TM concordance, the top matches are shown, with relevance decreasing further down.')
            st.write('You can select up to five documents for summarization.')
        st.session_state['quantile_cutoff'] = 0.75
        filtered_scored_docs = filter_by_quantile_session(st.session_state.get('scored_docs'))
        # ---- Add checkbox selection ----
        display_filtered_documents(filtered_scored_docs)
        
        #if 'weak_docs' in st.session_state and 'selected_docs' in st.session_state:
        #    specialized_regions()

# =========================
# 🔹 STEP 4: Summarize & Download
# =========================

if len(st.session_state.get("selected_docs",[])) > 0 and len(st.session_state.get("selected_docs",[])) < 6:

    st.markdown("### Summarize selected documents")
    st.write("You can generate summaries for the selected documents.")

    if st.button("📝 Generate summary"):
        with st.spinner("Generating summary... Please wait."):
            if st.session_state.get("selected_region"):
                specialized_regions()
            else:
                generalized_regions()
            summarize_documents()
        st.success("✅ Summaries generated successfully!")


    if 'summary_text' in st.session_state:
        col1, col2 = st.columns([1.2,1])
        with col1:
            st.write("#### Summary Report")
            st.write(st.session_state.get('summary_text'))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_file = summary_download()
            st.session_state["summary_file"] = summary_file
            st.download_button(
                label="⬇️ Download Summary",
                data=st.session_state.get("summary_file"),
                file_name=f"summary_report_{timestamp}.txt",
                mime="text/plain",
            )    
        with col2:
            summary_json = st.session_state.get("summary",[])
            
            if summary_json:
                region = st.session_state.get("selected_region")
                if region:
                    titles = [content["text"].split('\n')[0][2:-2] for code,content in summary_json.items()] #to remove the markdown formatting from the title

                    selected_title = st.selectbox(
                        "Select a topic for the map:",
                        titles,
                        key="map_topic_selector"
                    )

                    def categorize(nuts_id):
                        if nuts_id == source_id:
                            return 'Source'

                        if view_mode == "Top regions":
                            if nuts_id in relevant_ids:
                                return 'Relevant'
                            else:
                                return 'Other'

                        elif view_mode == "Closest regions":
                            if nuts_id in closest_ids:
                                return 'Closest'
                            else:
                                return 'Other'    
                    
                    view_mode = st.radio(
                                "Show regions:",
                                ["Top regions", "Closest regions"],
                                horizontal=True,
                                key="region_view_mode"
                            )

                    selected_doc = next(content for code,content in summary_json.items() if content["text"].split('\n')[0][2:-2] == selected_title)

                    relevant_ids = selected_doc["global_code"]
                    closest_ids = selected_doc["local_code"]

                    nuts2_meta = st.session_state.get("META_NUTS2_INDEX_KEY")
                    nuts2_codes = [entry['NUTS Code'] for entry in nuts2_meta]
                    
                    relevant_regions = {}
                    for entry in nuts2_meta:
                        if entry['NUTS Code'] in relevant_ids:
                            relevant_regions[entry['NUTS Code']] = entry['NUTS label'] 
                    relevant_regions_df = pd.DataFrame(list(relevant_regions.items()), columns=['NUTS_ID', 'Description'])
                    
                    closest_regions = {}
                    for entry in nuts2_meta:
                        if entry['NUTS Code'] in closest_ids:
                            closest_regions[entry['NUTS Code']] = entry['NUTS label'] 
                    closest_regions_df = pd.DataFrame(list(closest_regions.items()), columns=['NUTS_ID', 'Description'])

                    source_nuts_label = st.session_state.get("selected_region")
                    for entry in nuts2_meta:
                        if entry['NUTS label'] == source_nuts_label:
                            source_id = entry['NUTS Code']
                            source_label = entry['NUTS label']

                    nuts_gdf = st.session_state.get("RG_SHAPEFILE_NUTS2_KEY")
                    
                    nuts2_gdf = nuts_gdf[nuts_gdf['STAT_LEVL_'] == 2]
                    nuts2_gdf = nuts2_gdf[nuts2_gdf['NUTS_ID'].isin(nuts2_codes)]
                    nuts2_gdf['category'] = nuts2_gdf['NUTS_ID'].apply(categorize)
                    
                    nuts_lb = st.session_state.get("LB_SHAPEFILE_NUTS2_KEY")
                    pins_gdf = nuts_lb[
                    (nuts_lb["STAT_LEVL_"] == 2) &
                    (nuts_lb["NUTS_ID"].isin(nuts2_codes))
                ].copy()
                    pins_gdf['category'] = pins_gdf['NUTS_ID'].apply(categorize)

                    
                    if view_mode == "Top regions":
                        relevant_pins = pins_gdf[pins_gdf['category'] == 'Relevant']
                        #merge relevant_pins with relevant_regions_df to get the description for the hover text
                        relevant_pins = relevant_pins.merge(relevant_regions_df, left_on='NUTS_ID', right_on='NUTS_ID', how='left')
                        #include source pin in relevant pins for the hover text
                        source_pin = pins_gdf[pins_gdf['NUTS_ID'] == source_id]
                        source_pin = source_pin.merge(pd.DataFrame({'NUTS_ID':[source_id], 'Description':[source_label]}), on='NUTS_ID', how='left')
                        relevant_pins = pd.concat([relevant_pins, source_pin], ignore_index=True)

                    else:
                        closest_pins = pins_gdf[pins_gdf['category'] == 'Closest']
                        #merge closest_pins with closest_regions_df to get the description for the hover text
                        closest_pins = closest_pins.merge(closest_regions_df, left_on='NUTS_ID', right_on='NUTS_ID', how='left')
                        #include source pin in closest pins for the hover text
                        source_pin = pins_gdf[pins_gdf['NUTS_ID'] == source_id]
                        source_pin = source_pin.merge(pd.DataFrame({'NUTS_ID':[source_id], 'Description':[source_label]}), on='NUTS_ID', how='left')
                        closest_pins = pd.concat([closest_pins, source_pin], ignore_index=True)

             
                    # ---------------------------------------------------
                    # Build GeoJSON
                    # ---------------------------------------------------

                    geojson_data = json.loads(nuts2_gdf.to_json()) 

                    # ---------------------------------------------------
                    # Create map
                    # ---------------------------------------------------

                    fig = go.Figure()

                    # ---------------------------------------------------
                    # Add polygons
                    # ---------------------------------------------------

                    fig.add_trace(
                        go.Choropleth(
                            geojson=geojson_data,

                            featureidkey="properties.NUTS_ID",

                            locations=nuts2_gdf["NUTS_ID"],

                            z=[1] * len(nuts2_gdf),

                            colorscale=[
                                [0, "#e5e5e5"],
                                [1, "#e5e5e5"]
                            ],

                            showscale=False,

                            marker_line_color="black",
                            marker_line_width=0.6,

                            hovertext=nuts2_gdf["category"],

                            hovertemplate=
                                "<b>%{location}</b><br>%{hovertext}<extra></extra>"
                        )
                    )

                    # ---------------------------------------------------
                    # Add pins
                    # ---------------------------------------------------


                    if view_mode == "Top regions":
                        # ---------------------------------------------------
                        # Source pins
                        # ---------------------------------------------------

                        source_pins = relevant_pins[
                            relevant_pins["category"] == "Source"
                        ]

                        fig.add_trace(
                            go.Scattergeo(
                                lon=source_pins["LON"],
                                lat=source_pins["LAT"],

                                mode="markers",

                                text=source_pins["Description"],

                                hovertemplate=
                                    "<b>%{text}</b><extra></extra>",

                                marker=dict(
                                    size=16,
                                    color="blue",
                                    symbol="diamond"
                                ),

                                name="Selected region"
                            )
                        )

        
                        relevant_only = relevant_pins[
                            relevant_pins["category"] == "Relevant"
                        ]

                        fig.add_trace(
                            go.Scattergeo(
                                lon=relevant_only["LON"],
                                lat=relevant_only["LAT"],

                                mode="markers",

                                text=relevant_only["Description"],

                                hovertemplate=
                                    "<b>%{text}</b><extra></extra>",

                                marker=dict(
                                    size=14,
                                    color="red",
                                    symbol="diamond"
                                ),

                                name="Top regions"
                            )
                        )
                    else:
                        source_pins = closest_pins[closest_pins["category"] == "Source"]

                        fig.add_trace(
                            go.Scattergeo(
                                lon=source_pins["LON"],
                                lat=source_pins["LAT"],

                                mode="markers",

                                text=source_pins["Description"],

                                hovertemplate=
                                    "<b>%{text}</b><extra></extra>",

                                marker=dict(
                                    size=16,
                                    color="blue",
                                    symbol="diamond"
                                ),

                                name="Selected region"
                            )
                        )

                        # ---------------------------------------------------
                        # Closest pins
                        # ---------------------------------------------------

                        closest_only = closest_pins[closest_pins["category"] == "Closest"]

                        fig.add_trace(
                            go.Scattergeo(
                                lon=closest_only["LON"],
                                lat=closest_only["LAT"],

                                mode="markers",

                                text=closest_only["Description"],

                                hovertemplate=
                                    "<b>%{text}</b><extra></extra>",

                                marker=dict(
                                    size=14,
                                    color="green",
                                    symbol="diamond"
                                ),

                                name="Closest regions"
                            )
                        )
                    # ---------------------------------------------------
                    # Europe-only layout
                    # ---------------------------------------------------

                    fig.update_geos(
                        scope="europe",

                        projection_type="mercator",

                        showcountries=True,
                        countrycolor="white",

                        showcoastlines=True,
                        coastlinecolor="white",

                        showland=True,
                        landcolor="rgb(245,245,245)",

                        lataxis_range=[34, 72],
                        lonaxis_range=[-25, 45]
                    )

                    # ---------------------------------------------------
                    # Layout
                    # ---------------------------------------------------

                    fig.update_layout(
                        height=800,
                        width=800,

                        margin=dict(
                            l=0,
                            r=0,
                            t=0,
                            b=0
                        ),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.08,
                            xanchor="center",
                            x=0.5
                        )
                    )

                    # ---------------------------------------------------
                    # Show
                    # ---------------------------------------------------

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                if not region:
                    titles = [content["text"].split('\n')[0][2:-2] for code,content in summary_json.items()] #to remove the markdown formatting from the title

                    selected_title = st.selectbox(
                        "Select a topic for the map:",
                        titles,
                        key="map_topic_selector"
                    )

                    def categorize(nuts_id):
                        if nuts_id in relevant_ids:
                            return 'Relevant'
                        return 'Other'

                    selected_doc = next(content for code,content in summary_json.items() if content["text"].split('\n')[0][2:-2] == selected_title)
                    
                    relevant_ids = selected_doc["global_code"]
                    #build relevant_regions dictionary with keys ids and values text
                    
                    nuts2_meta = st.session_state.get("META_NUTS2_INDEX_KEY")
                    nuts2_codes = [entry['NUTS Code'] for entry in nuts2_meta]
                    relevant_regions = {}
                    for entry in nuts2_meta:
                        if entry['NUTS Code'] in relevant_ids:
                            relevant_regions[entry['NUTS Code']] = entry['NUTS label'] 
                    relevant_regions_df = pd.DataFrame(list(relevant_regions.items()), columns=['NUTS_ID', 'Description'])
                    
                    nuts_gdf = st.session_state.get("RG_SHAPEFILE_NUTS2_KEY")
                    
                    nuts2_gdf = nuts_gdf[nuts_gdf['STAT_LEVL_'] == 2]
                    nuts2_gdf = nuts2_gdf[nuts2_gdf['NUTS_ID'].isin(nuts2_codes)]

                    nuts2_gdf['category'] = nuts2_gdf['NUTS_ID'].apply(categorize)
                    
                    nuts_lb = st.session_state.get("LB_SHAPEFILE_NUTS2_KEY")
                    pins_gdf = nuts_lb[
                    (nuts_lb["STAT_LEVL_"] == 2) &
                    (nuts_lb["NUTS_ID"].isin(nuts2_codes))
                ].copy()
                    pins_gdf['category'] = pins_gdf['NUTS_ID'].apply(categorize)

                    relevant_pins = pins_gdf[pins_gdf['category'] == 'Relevant']
                    #merge relevant_pins with relevant_regions_df to get the description for the hover text
                    relevant_pins = relevant_pins.merge(relevant_regions_df, left_on='NUTS_ID', right_on='NUTS_ID', how='left')
            
                    # ---------------------------------------------------
                    # Build GeoJSON
                    # ---------------------------------------------------

                    geojson_data = json.loads(nuts2_gdf.to_json()) 

                    # ---------------------------------------------------
                    # Create map
                    # ---------------------------------------------------

                    fig = go.Figure()

                    # ---------------------------------------------------
                    # Add polygons
                    # ---------------------------------------------------

                    fig.add_trace(
                        go.Choropleth(
                            geojson=geojson_data,

                            featureidkey="properties.NUTS_ID",

                            locations=nuts2_gdf["NUTS_ID"],

                            z=[1] * len(nuts2_gdf),

                            colorscale=[
                                [0, "#e5e5e5"],
                                [1, "#e5e5e5"]
                            ],

                            showscale=False,

                            marker_line_color="black",
                            marker_line_width=0.6,

                            hovertext=nuts2_gdf["category"],

                            hovertemplate=
                                "<b>%{location}</b><br>%{hovertext}<extra></extra>"
                        )
                    )

                    # ---------------------------------------------------
                    # Add pins
                    # ---------------------------------------------------

                    fig.add_trace(
                        go.Scattergeo(
                            lon=relevant_pins["LON"],
                            lat=relevant_pins["LAT"],

                            mode="markers",

                            text=relevant_pins["Description"],

                            hovertemplate=
                                "<b>%{text}</b><extra></extra>",

                            marker=dict(
                                size=14,
                                color="red",
                                symbol="diamond"
                            ),

                            name="Relevant regions"
                        )
                    )

                    # ---------------------------------------------------
                    # Europe-only layout
                    # ---------------------------------------------------

                    fig.update_geos(
                        scope="europe",

                        projection_type="mercator",

                        showcountries=True,
                        countrycolor="white",

                        showcoastlines=True,
                        coastlinecolor="white",

                        showland=True,
                        landcolor="rgb(245,245,245)",

                        lataxis_range=[34, 72],
                        lonaxis_range=[-25, 45]
                    )

                    # ---------------------------------------------------
                    # Layout
                    # ---------------------------------------------------

                    fig.update_layout(
                        height=800,
                        width=800,

                        margin=dict(
                            l=0,
                            r=0,
                            t=0,
                            b=0
                        )
                    )

                    # ---------------------------------------------------
                    # Show
                    # ---------------------------------------------------

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )



            
# =========================
# 🔹 STEP 5: RATING
# =========================

st.write("#### Feedback")
st.write("Your feedback helps us improve the app! Please take a moment to share your thoughts. You will be taken to a Google Form where you can provide your feedback. Thank you!")
st.link_button(
    "Open feedback form",
    "https://docs.google.com/forms/d/e/1FAIpQLSdHgnvtR5buH767a_QgIt9ezYriQLWS9Ow2J-lJEdS6F-SO0Q/viewform?usp=sharing&ouid=109488069869282464948"
)

st.markdown("#### Restart Application")
# =========================
# 🔹 STEP 6: RESTART
# =========================
st.divider()
if st.button("Restart App"):
    # Clear all Streamlit session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("App has been restarted. Resetting all inputs...")
    st.rerun()



# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("#### Acknowledgements")
st.markdown("We thank the [Jantina Tammes School of Digital Society, Technology and AI](https://www.rug.nl/jantina-tammes-school/) at the University of Groningen for their support.")
st.caption("Powered by Streamlit | © 2025 Innovation App")