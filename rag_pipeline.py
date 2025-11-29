import os
import tempfile
from dotenv import load_dotenv # type: ignore

# --- LangChain Core Components ---
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain # type: ignore
from langchain_core.documents import Document

from typing import Optional

# Optional import for loading service-account credentials
try:
    from google.oauth2.service_account import Credentials as _ServiceAccountCredentials
except Exception:
    _ServiceAccountCredentials = None

# Google GenAI (langchain-google-genai) imports (optional)
try:
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    from langchain_google_genai.llms import ChatGoogleGenerativeAI
except Exception:
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

# --- Configuration ---
load_dotenv()
DB_PATH = "faiss_index"


def ensure_openai_api_key_loaded():
    """Ensure OPENAI_API_KEY is available in the environment.

    Order of precedence:
    1. Existing environment variable `OPENAI_API_KEY`
    2. Streamlit secrets `st.secrets['OPENAI_API_KEY']` (if running under Streamlit)
    3. `.env` file already loaded by `load_dotenv()`

    If found via Streamlit secrets, the function will set it into `os.environ`
    so downstream libraries that read `OPENAI_API_KEY` from the environment will work.
    Returns the key string or None if not found.
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    # Try Streamlit secrets if Streamlit is available
    try:
        import streamlit as _st # type: ignore
        key = _st.secrets.get("OPENAI_API_KEY") if hasattr(_st, "secrets") else None
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return key
    except Exception:
        # Not running in Streamlit or secrets not available
        pass

    # Final attempt: rely on dotenv having loaded .env earlier
    key = os.getenv("OPENAI_API_KEY")
    return key


def ensure_google_api_key_loaded():
    """Ensure GOOGLE_API_KEY is available in the environment.

    Tries env, Streamlit secrets, and .env via load_dotenv(). If found via
    Streamlit secrets, sets it into `os.environ` so downstream code can use it.
    Returns the key string or None if not found.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    try:
        import streamlit as _st # type: ignore
        key = _st.secrets.get("GOOGLE_API_KEY") if hasattr(_st, "secrets") else None
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            return key
    except Exception:
        pass

    key = os.getenv("GOOGLE_API_KEY")
    return key


def get_preferred_provider() -> Optional[str]:
    """Return the preferred provider: 'google' or 'openai' or None."""
    if ensure_google_api_key_loaded():
        return "google"
    if ensure_openai_api_key_loaded():
        return "openai"
    return None


def load_google_service_account_credentials():
    """If `GOOGLE_APPLICATION_CREDENTIALS` is set and the google oauth library
    is available, return a Credentials object. Otherwise return None.
    """
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not path or _ServiceAccountCredentials is None:
        return None

    try:
        creds = _ServiceAccountCredentials.from_service_account_file(path)
        return creds
    except Exception as e:
        print(f"Warning: failed to load service account credentials from {path}: {e}")
        return None

# --- 1. Data Ingestion Pipeline (The "Indexing" Phase) ---

def process_uploaded_files(uploaded_files: list) -> list[Document]:
    """
    Handles file upload, converts them to LangChain Documents, 
    and returns a list of all documents.
    """
    all_documents = []
    
    # Iterate through each uploaded file object
    for uploaded_file in uploaded_files:
        # Streamlit UploadedFile objects are file-like but don't have a path.
        # We must save them temporarily for PyPDFLoader, which needs a path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            # Write the bytes of the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Now, use the temporary path with PyPDFLoader
        try:
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            # Update metadata to show the original uploaded file name
            for doc in documents:
                doc.metadata['source'] = uploaded_file.name
            
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Crucially, delete the temporary file after processing
            os.remove(tmp_file_path)

    return all_documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Splits documents into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    return texts

# ... (rest of the file content)

def create_and_save_vector_store(texts: list[Document], db_path: str = DB_PATH):
    """Creates embeddings and builds/saves the FAISS vector store."""
    provider = get_preferred_provider()
    if provider is None:
        raise ValueError(
            "No API key found. Please set either GOOGLE_API_KEY or OPENAI_API_KEY in .env, Streamlit secrets, or environment."
        )

    # Try Google first if available, but fall back to OpenAI if Google fails at runtime
    embeddings = None
    used_provider = None

    if provider == "google":
        key = ensure_google_api_key_loaded()
        os.environ.setdefault("GOOGLE_API_KEY", key)
        if GoogleGenerativeAIEmbeddings is not None:
            try:
                # --- Prefer service account credentials when configured ---
                creds = load_google_service_account_credentials()
                if creds is not None:
                    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", credentials=creds)
                else:
                    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=key)
                used_provider = "google"
            except Exception as e:
                print(f"Warning: Google embeddings failed, will try OpenAI embeddings. Error: {e}")

    # If Google not used or failed, try OpenAI
    if embeddings is None:
        key = ensure_openai_api_key_loaded()
        if not key:
            raise ValueError(
                "No usable embeddings provider: Google failed or not configured, and OPENAI_API_KEY not found."
            )
        os.environ.setdefault("OPENAI_API_KEY", key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        used_provider = "openai"
    
    print(f"Creating FAISS index using {used_provider} embeddings...")
    vector_store = FAISS.from_documents(texts, embeddings)
    # Ensure the directory exists before saving
    os.makedirs(db_path, exist_ok=True)
    vector_store.save_local(db_path)
    print(f"Vector store saved to {db_path}")
    return vector_store

def load_vector_store(db_path: str = DB_PATH):
    """Loads an existing FAISS vector store."""
    provider = get_preferred_provider()
    # ... (API key loading logic remains the same) ...

    if provider == "google" and GoogleGenerativeAIEmbeddings is not None:
        try:
            key = ensure_google_api_key_loaded()
            os.environ.setdefault("GOOGLE_API_KEY", key)
            creds = load_google_service_account_credentials()
            if creds is not None:
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", credentials=creds)
            else:
                embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=key)
            used_provider = "google"
        except Exception as e:
            print(f"Warning: Google embeddings unavailable, falling back to OpenAI for loading vector store: {e}")
    # ... (OpenAI fallback logic remains the same) ...

    print(f"Loading FAISS index (expecting embeddings created with {used_provider})...")
    vector_store = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_store

# --- 2. RAG Chain Setup (Unchanged) ---

def setup_rag_chain(vector_store):
    """Sets up the Retrieval-Augmented Generation chain."""
    provider = get_preferred_provider()
    if provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed; cannot use Google LLM")
        # --- CORRECTED CHAT MODEL NAME ---
        creds = load_google_service_account_credentials()
        if creds is not None:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, credentials=creds)
        else:
            # fall back to passing API key if provided
            key = ensure_google_api_key_loaded()
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=key)
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # ... (rest of the function remains the same) ...

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful knowledge base agent. Use ONLY the following retrieved 
    context to answer the user's question. If you cannot find the answer in 
    the context, state clearly that the answer is not available in the documents.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain