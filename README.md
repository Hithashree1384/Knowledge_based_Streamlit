1. Overview of the Agent

This KnowledgeBase Agent is a Retrieval-Augmented Generation (RAG) system designed to answer specific questions based on a user's uploaded company documents (PDFs).
It provides grounding for the LLM's responses, ensuring answers are factual, traceable, and free from common hallucinations.

3. Features & Limitations

Section	Description
Features	:
Document Upload: Supports multiple PDF files via a Streamlit interface.

Semantic Search: Uses vector embeddings to find contextually relevant information.

Flexible Backend: Configured for high performance using Gemini-2.5-flash and Google Embeddings (with optional fallback to OpenAI).

Source Attribution: Provides the source file name and page number for every answer.


Limitations	
PDF Only: Currently supports only PDF files for ingestion.

Local Storage: Vector Index (FAISS) is stored locally on disk (faiss_index/) and must be rebuilt if documents change.

Single Turn: Does not maintain deep conversational history (each query is treated independently).


Export to Sheets
3. Tech Stack & APIs Used

Category	Component	Details

AI Models	Gemini-2.5-flash (LLM), Gemini-embedding-001 (Embeddings)	Used for answer generation and document vectorization.
Frameworks	LangChain	The primary orchestration framework for the RAG pipeline.
Vector DBs	FAISS	Used as the local, high-performance vector storage and retrieval mechanism.
UI	Streamlit	Provides the interactive web interface for file upload and chat.
APIs	Google GenAI API	Requires a valid GOOGLE_API_KEY for all model operations.
Export to Sheets


4. Setup & Run Instructions
To run this agent locally, follow these steps:

Clone the Repository:
Bash
git clone https://github.com/Hithashree1384/Knowledge_based_Streamlit.git

cd KnowledgeBase_Agent_RAG

Install Dependencies:
Bash
pip install -r requirements.txt

Set Up API Key:
Create a file named .env in the root directory.
Add your Google API key to the file:
GOOGLE_API_KEY=AIzaSyCr2RUwcl-dgteSJ3zc5yA9XAoQzODrgdI

Run the Application:
Bash
streamlit run app.py

Use the Agent: The app will open in your browser. Upload one or more PDF files in the sidebar and click "Build/Rebuild Knowledge Base (KB)". Once successful, you can ask questions in the chat interface.


5. Potential Improvements

Asynchronous Processing: Implement ingestion using asyncio to improve UI responsiveness during KB building.

Scalable Vector DB: Replace FAISS with a cloud Vector DB (like Pinecone or Weaviate) for persistence and multi-user support.

Structured Output: Use LangChain's structured output capability to extract key facts from the documents in JSON format.

Advanced RAG: Implement advanced techniques like Reranking or Self-Querying for improved retrieval accuracy.
