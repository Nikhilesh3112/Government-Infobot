# Government Infobot

A Streamlit app that answers questions about Indian government schemes using a Retrieval-Augmented Generation (RAG) pipeline. It queries a prebuilt FAISS knowledge base created from:

- PDF(s) in `pdf/`
- A curated web page (Wikipedia list of Indian government schemes)

The app uses OpenAI for embeddings and chat responses, with a fallback that returns snippet-only answers if the model is unavailable.

## Features
- Login/Register using `streamlit-authenticator`
- Retrieves from merged FAISS stores: `faiss_pdf_1` and `faiss_url_1`
- Chat UI with short, context-aware answers
- Free-tier friendly: low `k`, trimmed context, small responses
- Fallback to snippet extract when API quota/errors occur

## Requirements
- Python 3.11

## Setup
1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure OpenAI credentials (pick one)
- Option A: Environment variables
  ```bash
  export OPENAI_API_KEY="sk-REPLACE_ME"
  export OPENAI_CHAT_MODEL="gpt-4o-mini"
  export OPENAI_EMBED_MODEL="text-embedding-3-small"
  ```
- Option B: Streamlit secrets (local only; keep untracked by git)
  Create `.streamlit/secrets.toml` with:
  ```toml
  [openai]
  api_key = "sk-REPLACE_ME"
  chat_model = "gpt-4o-mini"
  embed_model = "text-embedding-3-small"
  ```
  Ensure `.streamlit/secrets.toml` is listed in `.gitignore`.

3) (Optional) Update authentication users in `config.yaml` under `credentials` and `preauthorized`.

4) Run the app
```bash
streamlit run main.py
```

## Usage
- Login or register via the UI
- Ask a question about government schemes; the app retrieves from the FAISS stores and responds

## Data/Indexes
- `faiss_pdf_1/` and `faiss_url_1/` contain the prebuilt FAISS indexes used at runtime
- To (re)build indexes, see `loader.py` for simple scripts to create FAISS from a URL or PDF directory

## Notes
- Do not commit API keys or secrets. Keep `.streamlit/secrets.toml` untracked and rotate any exposed keys.
