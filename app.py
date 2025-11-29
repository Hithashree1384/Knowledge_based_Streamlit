import streamlit as st # type: ignore
import os
from rag_pipeline import (
    process_uploaded_files, # Modified function name
    split_documents, 
    create_and_save_vector_store, 
    load_vector_store, 
    setup_rag_chain,
    DB_PATH
)

# If deployed on Streamlit Cloud, allow providing the Google service-account JSON
# via `st.secrets['GOOGLE_SERVICE_ACCOUNT']`. If present, write it to a
# temporary file and set `GOOGLE_APPLICATION_CREDENTIALS` so Google client
# libraries pick it up for ADC.
try:
    if hasattr(st, "secrets") and "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
        import json

        sa = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
        sa_json = sa if isinstance(sa, str) else json.dumps(sa)
        sa_path = "/tmp/service-account.json"
        with open(sa_path, "w", encoding="utf-8") as _f:
            _f.write(sa_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
        st.info("Loaded Google service account from Streamlit secrets")
except Exception as _e:
    # Non-fatal; continue without ADC if not provided
    try:
        st.warning(f"Could not load Google service account from secrets: {_e}")
    except Exception:
        pass

st.set_page_config(page_title="KnowledgeBase Agent", layout="wide")
st.title("📚 Company KnowledgeBase RAG Agent")

# Ensure session state keys exist early (prevents missing-attribute errors)
st.session_state.setdefault('messages', [])
st.session_state.setdefault('rag_chain', None)

# --- Session State and Initialization ---

@st.cache_resource
def initialize_agent():
    """Initializes and caches the RAG Agent components."""
    if os.path.exists(DB_PATH):
        try:
            vector_store = load_vector_store()
            rag_chain = setup_rag_chain(vector_store)
            st.session_state.rag_chain = rag_chain
            st.success("Existing knowledge base loaded successfully!")
        except Exception as e:
            st.error(f"Error loading existing vector store: {e}. Please rebuild.")
    
    # safe defaults in case session state isn't populated yet
    st.session_state.setdefault('messages', [])
    st.session_state.setdefault('rag_chain', None)

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("1. Document Ingestion")
    
    # NEW: Use st.file_uploader to allow browsing/uploading
    uploaded_files = st.file_uploader(
        "Upload PDF Files to Build KB",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("Build/Rebuild Knowledge Base (KB)"):
        if not uploaded_files:
            st.warning("Please upload one or more PDF files first.")
        else:
            with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                try:
                    # 1. Process files (temp save, load, delete)
                    documents = process_uploaded_files(uploaded_files)
                    
                    # 2. Split and Vectorize
                    texts = split_documents(documents)
                    vector_store = create_and_save_vector_store(texts)
                    
                    # 3. Setup Agent
                    st.session_state.rag_chain = setup_rag_chain(vector_store)
                    
                    st.success(f"Knowledge Base built from {len(uploaded_files)} file(s) and ready to use!")
                    st.session_state.messages = [] # Clear chat history on rebuild
                except Exception as e:
                    err_msg = str(e)
                    st.error(f"An error occurred during KB creation: {err_msg}")

                    # Provide actionable next steps for common Google / OpenAI issues
                    if "models/" in err_msg or "DefaultCredentialsError" in err_msg or "Google" in err_msg:
                        st.info(
                            "Google embeddings failed. Options:\n"
                            "1) Add `OPENAI_API_KEY` to `.env` (or environment) to use OpenAI embeddings as a fallback.\n"
                            "2) Configure Google Application Default Credentials (set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON) and ensure the account has access to embedding-capable models.\n"
                            "3) Remove `GOOGLE_API_KEY` from `.env` to force the app to choose OpenAI if `OPENAI_API_KEY` is present."
                        )
                    else:
                        st.info("Check your API keys and try again. Ensure `OPENAI_API_KEY` is set for OpenAI usage, or configure Google credentials correctly.")

    st.markdown("---")
    st.header("2. Reset")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---

initialize_agent()

# Display chat messages from history (use .get for safety)
for message in st.session_state.get('messages', []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    
    st.session_state.setdefault('messages', []).append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chain is None:
        st.warning("The RAG Agent is not initialized. Please upload files and click 'Build KB' first.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            response = st.session_state.rag_chain.invoke({"input": prompt})
            
            answer = response['answer']
            sources = "\n".join([
                f"- **Source:** {doc.metadata.get('source', 'Unknown Source')} (Page: {doc.metadata.get('page', 'N/A')})"
                for doc in response['context']
            ])
            
            full_response = f"{answer}\n\n---\n\n**Sources Used:**\n{sources}"
            
            st.markdown(full_response)
            st.session_state.setdefault('messages', []).append({"role": "assistant", "content": full_response})