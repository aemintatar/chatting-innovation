import streamlit as st
from settings import *
from innovation_tools import *
from datetime import datetime

# =========================
# 🔧 PAGE CONFIG
# =========================
st.set_page_config(page_title="Innovation Application", layout="wide") # Configures the browser tab title and page layout.
st.title("Innovation Assistant") # Main title of the app.
st.markdown("""
            This application (powered by Google ADK, Mistral, and Streamlit) is based on the work of Abbasiharofteh, Castaldi, and Petralia (forthcoming),
            which establishes a concordance between patents and trademarks. You can enter a *technology* to discover the *goods* and *services* it enables, 
            or input a *good* or a *service* to identify the *technologies* required for its development and market introduction. The results include a summary 
            of the goods/services or technologies, along with the strength of their associations (quantiles) derived from the patent-to-trademark concordance.
            Note: More specific queries lead to better results. For example, instead of entering “Drones,” a more focused query like “Drone Power System” or “Drone Flight Controller” can yield more accurate and meaningful associations.
            """)
                    
# Descriptive text.
st.divider() # A visual separator.



# data (main frame)
st.markdown("#### Data Management")
if st.button("Load Data"):
    with st.spinner("Loading document indexes..."):
        # Load FAISS indices
        META_ALL_INDEX_PATH = './output/allmetadata.pkl'

        FAISS_TECH_INDEX_PATH = './output/faiss_tech_index.bin'
        META_TECH_INDEX_PATH = './output/techmetadata.pkl'

        FAISS_SERVICE_INDEX_PATH = './output/faiss_service_index.bin'
        META_SERVICE_INDEX_PATH = './output/servicemetadata.pkl'

        FAISS_GOOD_INDEX_PATH = './output/faiss_good_index.bin'
        META_GOOD_INDEX_PATH = './output/goodmetadata.pkl'

        FAISS_MARKET_LQ_INDEX_PATH = './output/faiss_market_lq_index.bin'
        META_MARKET_LQ_INDEX_PATH = './output/market_lq_metadata.pkl'

        FAISS_TECH_LQ_INDEX_PATH = './output/faiss_tech_lq_index.bin'
        META_TECH_LQ_INDEX_PATH = './output/tech_lq_metadata.pkl'

        FAISS_DISTANCE_INDEX_PATH = './output/faiss_dist_index.bin'
        META_NUTS2_INDEX_PATH = './output/nuts2.pkl'

        all_meta = load_meta(META_ALL_INDEX_PATH)

        tech_index = load_index(FAISS_TECH_INDEX_PATH)
        tech_meta = load_meta(META_TECH_INDEX_PATH)
        
        service_index = load_index(FAISS_SERVICE_INDEX_PATH)
        service_meta = load_meta(META_SERVICE_INDEX_PATH)
        
        good_index = load_index(FAISS_GOOD_INDEX_PATH)
        good_meta = load_meta(META_GOOD_INDEX_PATH)
        
        market_lq_index = load_index(index_path=FAISS_MARKET_LQ_INDEX_PATH)
        market_lq_meta = load_meta(META_MARKET_LQ_INDEX_PATH)
        
        tech_lq_index = load_index(index_path=FAISS_TECH_LQ_INDEX_PATH)
        tech_lq_meta = load_meta(META_TECH_LQ_INDEX_PATH)
        
        distance_index = load_index(index_path=FAISS_DISTANCE_INDEX_PATH)
        nuts2_meta = load_meta(metadata_path=META_NUTS2_INDEX_PATH)
        st.session_state[META_ALL_INDEX_KEY] = all_meta

        st.session_state[FAISS_TECH_INDEX_KEY] = tech_index
        st.session_state[META_TECH_INDEX_KEY] = tech_meta

        st.session_state[FAISS_SERVICE_INDEX_KEY] = service_index
        st.session_state[META_SERVICE_INDEX_KEY] = service_meta

        st.session_state[FAISS_GOOD_INDEX_KEY] = good_index
        st.session_state[META_GOOD_INDEX_KEY] = good_meta

        st.session_state[FAISS_MARKET_LQ_INDEX_KEY] = market_lq_index
        st.session_state[META_MARKET_LQ_INDEX_KEY] = market_lq_meta
        
        st.session_state[FAISS_TECH_LQ_INDEX_KEY] = tech_lq_index
        st.session_state[META_TECH_LQ_INDEX_KEY] = tech_lq_meta
        
        st.session_state[FAISS_DISTANCE_INDEX_KEY] = distance_index
        st.session_state[META_NUTS2_INDEX_KEY] = nuts2_meta
        st.success("Data Loaded!")

st.divider() # A visual separator.
st.markdown("#### Search Parameters")
#  Topic / country / region selectors (main frame)

col1, col2, col3 = st.columns(3)

with col1:
    topic_value = st.selectbox(
    "Topic",
    ["-- None --", "Technology", "Service", "Good"],
    index=0,
    key="sidebar_topic"
)
with col2:
    META_NUTS2_INDEX_PATH = './output/nuts2.pkl'
    nuts2_meta = load_meta(metadata_path=META_NUTS2_INDEX_PATH)
    
    country_regions = load_country_regions(nuts2_meta)
    country_options = ["-- None --"] + sorted(list(country_regions.keys()))
    country_value = st.selectbox("Country (optional)", 
                           country_options, 
                           index=0,
                           key="sidebar_country")
with col3:
    region_options = ["-- None --"]
    if country_value and not country_value.startswith("--"):
        region_options += sorted(country_regions.get(country_value, []))
    
    region_value = st.selectbox("Region (optional)", 
                          region_options, 
                          index=0,
                          key="sidebar_region")
# Apply context button
if st.button("Apply Parameters"):
    st.session_state["detected_topic"] = None if topic_value.startswith("--") else topic_value
    st.session_state["country_code"] = None if country_value.startswith("--") else country_value
    st.session_state["selected_region"] = None if region_value.startswith("--") else region_value
    st.success("Parameters applied!")

st.divider() # A visual separator.
st.markdown("#### Query")
prompt = st.text_area("Enter your product or technology idea:")

if MESSAGE_HISTORY_KEY not in st.session_state:
    st.session_state[MESSAGE_HISTORY_KEY] = []

for message in st.session_state[MESSAGE_HISTORY_KEY]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# 🔹 STEP 1: Interpret and retrieve 
# =========================
if st.button("🔍 Retrieve Documents"):
    if not st.session_state.get("detected_topic"):
        st.warning("⚠️ Please select a topic first.")
    else:
        query = st.session_state.get("user_idea_input", "")
        retrieve_documents(query)  # your function saves to st.session_state['retrieved documents']
        st.success(f"✅ {len(st.session_state['retrieved documents'])} documents retrieved.")

# =========================
# 🔹 STEP 2: Display & Select
# =========================
display_retrieved_documents()

# =========================
# 🔹 STEP 3: Scoring
# =========================
if st.session_state.get('selected_codes'):
    if st.button("📊 Score Selected Documents"):
        scored_docs, text_for_summary = scoring_documents()  # scoring_tool saves results to session_state['text_to_summarize']
        st.session_state['text_to_summarize'] = text_for_summary
        st.dataframe(scored_docs)  # display scored docs as table
        st.success("✅ Documents scored successfully!")

# =========================
# 🔹 STEP 4: Summarize & Download
# =========================
if "text_to_summarize" in st.session_state and st.session_state["text_to_summarize"]:
    if st.button("🧾 Summarize"):
        summary, summary_file = summarize_documents(st.session_state["text_to_summarize"])
        st.session_state["summary"] = summary
        st.success("Summary generated successfully!")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="⬇️ Download Summary",
            data=summary_file,
            file_name=f"summary_report_{timestamp}.txt",
            mime="text/plain",
        )

st.divider()
st.markdown("#### 🔁 Restart Application")

if st.button("Restart App"):
    # Clear all Streamlit session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Clear cached resources or data if desired
    # st.cache_data.clear()
    # st.cache_resource.clear()

    # Force rerun to redraw UI elements with defaults
    st.success("App has been restarted. Resetting all inputs...")
    st.rerun()



# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Powered by Streamlit | © 2025 Innovation Assistant")