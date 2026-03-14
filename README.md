# Government Infobot

A Streamlit app that answers questions about Indian government schemes using a Retrieval-Augmented Generation (RAG) pipeline. It queries a prebuilt FAISS knowledge base created from:

- PDF(s) in `pdf/`
- A curated web page (Wikipedia list of Indian government schemes)

The app uses **Google Gemini** for both embeddings (`gemini-embedding-001`) and chat responses (`gemini-2.5-flash`), with built-in rate-limit handling and intelligent fallbacks to general knowledge if the documents lack the needed information.

## Features
- Complete migration to **Google Gemini API** (No OpenAI dependencies)
- Login/Register using `streamlit-authenticator` (fixed seamless one-click login routing)
- Retrieves from merged FAISS stores: `faiss_pdf_1` and `faiss_url_1`
- Chat UI with brief, multilingual, context-aware answers
- Free-tier friendly with targeted rate-limit error messaging (HTTP 429 warnings)

## Requirements
- Python 3.10+

## Setup
1) Create and activate a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
```

2) Configure Google Gemini credentials
- Option A: Environment variables
  ```bash
  export GOOGLE_API_KEY="AIzaSy..."
  export GOOGLE_CHAT_MODEL="gemini-2.5-flash"
  export GOOGLE_EMBED_MODEL="models/gemini-embedding-001"
  ```
- Option B: Streamlit secrets (local only)
  Create `.streamlit/secrets.toml`:
  ```toml
  [google]
  api_key = "AIzaSy..."
  chat_model = "gemini-2.5-flash"
  embed_model = "models/gemini-embedding-001"
  ```
  *(Ensure `.streamlit/secrets.toml` is ignored in git)*

3) Run the app
```bash
streamlit run main.py
```

## Data/Indexes
- `faiss_pdf_1/` and `faiss_url_1/` contain the prebuilt FAISS indexes used at runtime.
- To rebuild indexes (if adding new PDFs/URLs), run `python loader.py`. Note that rebuilding requires hitting the embedding API.

## Notes
- **Security:** Do not commit API keys or secrets. Keep `.streamlit/secrets.toml` untracked. User credentials registered in the app are automatically hashed and saved to `config.yaml`.
