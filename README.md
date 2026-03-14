# Government Infobot

A powerful, multilingual Streamlit web application that answers questions about Indian Government Schemes. The app leverages cutting-edge **Natural Language Processing (NLP)**, **LangChain**, and **Retrieval-Augmented Generation (RAG)** to provide accurate, concise, and context-aware responses.

## How It Works
The app acts as a smart, conversational "Government Infobot." It ingests data from two primary sources:
1. **Local Documents:** PDFs stored in the `pdf/` directory (e.g., scheme documents).
2. **Web Scraping:** A curated Wikipedia page listing all Indian government schemes.

Using **Google Gemini's Embedding Models**, the app converts these documents into mathematical vectors and stores them in a highly efficient **FAISS** vector database. When a user asks a question, the app's NLP engine performs a semantic search to retrieve the most relevant paragraphs from the database. It then feeds this context to the **Google Gemini 2.5 Flash** Large Language Model (LLM) to generate a helpful, human-like response in the user's native language. If the answer isn't in the context, it smartly falls back to its general knowledge.

## Key Features
- **Advanced NLP & RAG:** Built with LangChain to seamlessly connect the LLM with the FAISS vector databases.
- **Multilingual Support:** Automatically detects the user's language and replies in the same language.
- **Secure Authentication:** Built-in seamless login, registration, and secure cookie management using `streamlit-authenticator`. 
- **Privacy First:** Chat histories are strictly isolated per user and wiped upon logout.
- **API Rate Limiting:** Gracefully handles API exhaustion with clear UI warnings (HTTP 429 support).

## Tech Stack
- **Frontend & UI:** Streamlit
- **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash` and `gemini-embedding-001`)
- **Orchestration:** LangChain
- **Vector Database:** FAISS
- **Authentication:** Streamlit-Authenticator

## Local Setup Instructions

1) **Create and activate a virtual environment:**
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
```

2) **Configure Google Gemini API Credentials:**
Provide your API key to Streamlit secrets by creating a `.streamlit/secrets.toml` file:
```toml
[google]
api_key = "AIzaSy..."
chat_model = "gemini-2.5-flash"
embed_model = "models/gemini-embedding-001"
```
*(Note: `.streamlit/secrets.toml` is ignored by git to protect your keys).*

3) **Run the Application:**
```bash
streamlit run main.py
```

## Data Management & Rebuilding Indexes
- The `faiss_pdf_1/` and `faiss_url_1/` folders contain the prebuilt FAISS indexes.
- If you add new PDFs to the `pdf/` folder or wish to scrape new URLs, you can run the provided `loader.py` script to rebuild the vector databases.
