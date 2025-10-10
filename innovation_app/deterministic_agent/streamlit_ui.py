import os
import gdown
import streamlit as st
from settings import *
from datetime import datetime
from innovation_tools import *

# =========================
# 🔧 PAGE CONFIG
# =========================
st.set_page_config(page_title="Innovation Application", layout="wide") # Configures the browser tab title and page layout.
st.title("Innovation Assistant") # Main title of the app.
st.markdown("""
            This application (powered by Mistral and Streamlit) is based on the work of Abbasiharofteh, Castaldi, and Petralia (forthcoming),
            which establishes a concordance between patents and trademarks. You can enter a *technology* to discover the *goods* and *services* it enables, 
            or input a *good* or a *service* to identify the *technologies* required for its development and market introduction. The results include a summary 
            of the goods/services or technologies, along with the strength of their associations (quantiles) derived from the patent-to-trademark concordance.
            Note: More specific queries lead to better results. For example, instead of entering “Drones,” a more focused query like “Drone Power System” or “Drone Flight Controller” can yield more accurate and meaningful associations.
            """)
                    

# 🔁 Cached data loader (runs only once per session)
@st.cache_resource
def load_all_data_from_drive():
    st.write("📦 Loading document indexes...")

    # 🔹 Google Drive file IDs (replace with your real ones)
    DRIVE_FILES = {
        "META_ALL_INDEX_PATH": "1dIGLD4JLS8Jf_LKH-bglwPGF4-HA0eKs",  # allmetadata.pkl
        "FAISS_TECH_INDEX_PATH": "1dVVJo7gQNVxuIEfP-Fo9g8_yFTAQ2037",
        "META_TECH_INDEX_PATH": "17l9z2n6gmYT7RVi40NI99-NFq5OwyABE",
        "FAISS_SERVICE_INDEX_PATH": "1yXmOlTKYbws0gPfxJ_3PtPz3jy3bhhpD",
        "META_SERVICE_INDEX_PATH": "11_U_y1E4cwSqh1jYEXOUwEZoKa022BEA",
        "FAISS_GOOD_INDEX_PATH": "1nQNbPvYH9hCguLuJpodnWKBI2GRXNttD",
        "META_GOOD_INDEX_PATH": "11J-wahQa8qoKkm7FKE4rBNWJsYQecCjY",
        "FAISS_MARKET_LQ_INDEX_PATH": "1Ur0dl3407o2h1uQyRis8ggtFurV1Nzk7",
        "META_MARKET_LQ_INDEX_PATH": "1DypEu3UQUAlbDaM90tER8rsTLgnxxIiz",
        "FAISS_TECH_LQ_INDEX_PATH": "11wCzPhGbRTjZRQ_SSkpaVJw_Q3e-hpAF",
        "META_TECH_LQ_INDEX_PATH": "1cDuXTx34MxvwoUpif8YDyrrAhVretLg3",
        "FAISS_DISTANCE_INDEX_PATH": "1qsDZVwDFIXoRLhOUeXqAUPIZKutyaRBt",
        "META_NUTS2_INDEX_PATH": "14xU2zeR1fhajs5wot7W80JVYRY7_vKT8",
    }

    data_dir = "/tmp/data"
    os.makedirs(data_dir, exist_ok=True)
    paths = {}

    # 🔹 Download missing files only
    for name, file_id in DRIVE_FILES.items():
        dest = os.path.join(data_dir, name + os.path.splitext(file_id)[0] + ".bin")  # unique filename
        url = f"https://drive.google.com/uc?id={file_id}"

        if not os.path.exists(dest):
            st.write(f"⬇️ Downloading {name}...")
            gdown.download(url, dest, quiet=False)

        paths[name] = dest

    # 🔹 Load data using your existing functions
    st.write("⚙️ Reading data files into memory...")
    all_meta = load_meta(paths["META_ALL_INDEX_PATH"])
    tech_index = load_index(paths["FAISS_TECH_INDEX_PATH"])
    tech_meta = load_meta(paths["META_TECH_INDEX_PATH"])
    service_index = load_index(paths["FAISS_SERVICE_INDEX_PATH"])
    service_meta = load_meta(paths["META_SERVICE_INDEX_PATH"])
    good_index = load_index(paths["FAISS_GOOD_INDEX_PATH"])
    good_meta = load_meta(paths["META_GOOD_INDEX_PATH"])
    market_lq_index = load_index(paths["FAISS_MARKET_LQ_INDEX_PATH"])
    market_lq_meta = load_meta(paths["META_MARKET_LQ_INDEX_PATH"])
    tech_lq_index = load_index(paths["FAISS_TECH_LQ_INDEX_PATH"])
    tech_lq_meta = load_meta(paths["META_TECH_LQ_INDEX_PATH"])
    distance_index = load_index(paths["FAISS_DISTANCE_INDEX_PATH"])
    nuts2_meta = load_meta(paths["META_NUTS2_INDEX_PATH"])

    # Return all data as a dictionary
    return {
        "META_ALL_INDEX_KEY": all_meta,
        "FAISS_TECH_INDEX_KEY": tech_index,
        "META_TECH_INDEX_KEY": tech_meta,
        "FAISS_SERVICE_INDEX_KEY": service_index,
        "META_SERVICE_INDEX_KEY": service_meta,
        "FAISS_GOOD_INDEX_KEY": good_index,
        "META_GOOD_INDEX_KEY": good_meta,
        "FAISS_MARKET_LQ_INDEX_KEY": market_lq_index,
        "META_MARKET_LQ_INDEX_KEY": market_lq_meta,
        "FAISS_TECH_LQ_INDEX_KEY": tech_lq_index,
        "META_TECH_LQ_INDEX_KEY": tech_lq_meta,
        "FAISS_DISTANCE_INDEX_KEY": distance_index,
        "META_NUTS2_INDEX_KEY": nuts2_meta,
    }

# 🧠 Initialize data once
if "data_loaded" not in st.session_state:
    with st.spinner("Loading all data from Google Drive..."):
        data_dict = load_all_data_from_drive()
        for k, v in data_dict.items():
            st.session_state[k] = v
        st.session_state["data_loaded"] = True
        st.success("✅ Data loaded successfully!")


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
    nuts2_meta = st.session_state.get("META_NUTS2_INDEX_KEY")

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
prompt = st.text_area("Enter your product or technology idea:",key="user_idea_input")

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
    elif st.session_state.get("country_code") and not st.session_state.get("selected_region"):
        st.warning("⚠️ Please select a region as well!")
    else:
        query = st.session_state.get("user_idea_input")
        if not query.strip():
            st.warning("⚠️ Please enter an idea before retrieving documents.")
        else:
            retrieve_documents(query)
            docs = st.session_state.get('retrieved documents', [])
            st.success(f"✅ {len(docs)} documents retrieved.")


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
st.markdown("#### Restart Application")

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