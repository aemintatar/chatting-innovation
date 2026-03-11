import os
import gdown
import streamlit as st
from settings import *
from datetime import datetime
from innovation_tools import *
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
        "META_NUTS2_INDEX_PATH": "nuts2.pkl"
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
st.session_state["selected_region_index"] = 0  # Initialize selected region index in session state
st.session_state["selected_country_index"] = 0  # Initialize selected country index in session state
st.session_state["selected_context_index"] = 0  # Initialize selected context index in session state
with col1:
    context_value = st.selectbox(
    "**Context** (required)",["-- None --", "Technology", "Service", "Good"],
    index=st.session_state.get("selected_context_index", 0),
    key="sidebar_context"
)
with col2:
    nuts2_meta = st.session_state.get("META_NUTS2_INDEX_KEY")
    country_regions = load_country_regions(nuts2_meta)[0]
    country_names = load_country_regions(nuts2_meta)[1]
    country_options = ["-- None --"] + country_names
    country_value = st.selectbox("**Country** (optional)", 
                           country_options, 
                           index=st.session_state.get("selected_country_index", 0),
                           key="sidebar_country")
with col3:
    region_options = ["-- None --"]
    if country_value and not country_value.startswith("--"):
        country_code = country_value.split("(")[-1].split(")")[0].strip()
        region_options += sorted(country_regions.get(country_code, []))
    region_options = [option for option in region_options if option!='Extra-Regio NUTS 2']
    region_value = st.selectbox("**Region** (optional)", 
                          region_options, 
                          index=st.session_state.get("selected_region_index", 0),
                          key="sidebar_region")
# Apply context button
if st.button("Apply Parameters"):
    st.session_state["detected_context"] = None if context_value.startswith("--") else context_value.lower()
    st.session_state["country_code"] = None if country_value.startswith("--") else country_value
    st.session_state["selected_region"] = None if region_value.startswith("--") else region_value
    st.success("Parameters applied!")

if 'detected_context' in st.session_state:
    st.write('#### Specializations')
    df = get_top_lq()
    st.session_state['specialization'] = df
    st.dataframe(st.session_state.get('specialization'))
    

st.divider() # A visual separator.
st.markdown("#### Query")
prompt = st.text_area("",key="query",placeholder="Enter your product or technology idea.")


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
        st.write('Based on the PAT2TM concordance, we show the closest matches first. The further down you go, the less related they are to your question.')
        st.write('Up to five items can be selected')
        st.session_state['quantile_cutoff'] = 0.75
        filtered_scored_docs = filter_by_quantile_session(st.session_state.get('scored_docs'))
        # ---- Add checkbox selection ----
        display_filtered_documents(filtered_scored_docs)
        if 'low_lq' in st.session_state:
            specialized_regions()


# =========================
# 🔹 STEP 4: Summarize & Download
# =========================
selected_df = st.session_state.get("selected_docs")
if len(selected_df) > 0:
    st.markdown("### Summarize selected documents")
    st.write("You can generate summaries for the selected documents.")

    if st.button("📝 Generate summaries"):
        summary = summarize_documents()
        st.session_state["summary"] = summary
        st.success("✅ Summaries generated successfully!")

    if 'summary' in st.session_state:
        st.write("#### Summary")
        st.write(st.session_state.get('summary'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_download()
        st.session_state["summary_file"] = summary_file
        st.download_button(
            label="⬇️ Download Summary",
            data=st.session_state.get("summary_file"),
            file_name=f"summary_report_{timestamp}.txt",
            mime="text/plain",
        )    

st.divider()

# =========================
# RATING
# =========================
st.write("#### Feedback")
st.write("Your feedback helps us improve the app! Please take a moment to share your thoughts. We are not collecting any personally identifiable information, and all feedback is anonymous.")
if st.button("Give feedback"):
    st.session_state["show_feedback"] = True
    st.markdown("#### Rate this App")

    expectation = st.radio(
        "Did this output meet your expectations?",
        options=["Yes", "Partially", "No"],
        index=None
    )

    useful = st.radio(
        "Was this result useful for your task?",
        options=["Yes", "Partially", "No"],
        index=None
    )

    practical = st.radio(
        "Was this app confusing to use?",
        options=["Yes", "Partially", "No"],
        index=None
    )

    apprating = st.radio(
        "Overall rating",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: "⭐" * x,
        horizontal=True
    )

    comments = st.text_area(
        "Additional comments (optional)",
        placeholder="Tell us how this could be improved..."
    )



    if st.button("Submit feedback"):
        
        feedback = {
            "timestamp": datetime.now(),
            "apprating": apprating,
            "met_expectations": expectation,
            "useful_for_task": useful,
            "comments": comments,
            "practical_to_use": practical
        }

        feedback_df = pd.DataFrame([feedback])
        st.session_state["feedback_file_exists"] = True

        if "feedback_file_exists" in st.session_state and st.session_state["feedback_file_exists"]:
            with open("feedback_log.csv", "rb") as f:
                st.download_button(
                    label="📥 Download feedback CSV",
                    data=f,
                    file_name="feedback_log.csv",
                    mime="text/csv"
                )

        

        st.success("✅ Thank you for your feedback!")

st.markdown("#### Restart Application")
# =========================
# RESTART
# =========================
st.divider()
if st.button("Restart App"):
    # Clear all Streamlit session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state["selected_region_index"] = 0  # Initialize selected region index in session state
    st.session_state["selected_country_index"] = 0  # Initialize selected country index in session state
    st.session_state["selected_context_index"] = 0
    st.success("App has been restarted. Resetting all inputs...")
    st.rerun()



# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("#### Acknowledgements")
st.markdown("We thank the [Jantina Tammes School of Digital Society, Technology and AI](https://www.rug.nl/jantina-tammes-school/) at the University of Groningen for their support.")
st.caption("Powered by Streamlit | © 2025 Innovation App")