# Infobot (Government Infobot)

A Streamlit RAG app that answers questions using a prebuilt FAISS knowledge base from:
- PDF(s) in `pdf/`
- A curated web source (Wikipedia list of Indian government schemes)

It uses OpenAI for chat responses and embeddings, with a graceful fallback to snippet-only answers if the model is unavailable or quota is exceeded.

## Features
- Login/Register via streamlit-authenticator
- Retrieves from merged FAISS stores (`faiss_pdf_1`, `faiss_url_1`)
- Chat with context-aware answers
- Free-tier friendly: low `k`, trimmed context, short responses
- Fallback: snippet-only extract when API quota/errors occur

## Requirements
- Python 3.11

## Setup
1) Create and activate a virtual environment
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Provide your OpenAI API key (choose one):
- Option A: Streamlit secrets (recommended; not tracked by git)
  - Create `.streamlit/secrets.toml`:
    ```toml
    [openai]
    api_key = "sk-REPLACE_ME"
    chat_model = "gpt-4o-mini"
    embed_model = "text-embedding-3-small"
    ```
- Option B: Environment variables (in same shell before running)
    ```bash
    export OPENAI_API_KEY="sk-REPLACE_ME"
    export OPENAI_CHAT_MODEL="gpt-4o-mini"
    export OPENAI_EMBED_MODEL="text-embedding-3-small"
    ```
- Option C: config.yaml (not recommended for public repos)
  - Add an `openai` block with the same three keys above.

3) Run the app
```bash
source .venv/bin/activate
streamlit run main.py
```

## Usage
- Open the Streamlit URL shown in the terminal.
- Register or login.
- Ask questions like:
  - "List key schemes for farmers mentioned in the PDF."
  - "Summarize education-related benefits mentioned in the PDF."

## Snippet-only fallback
- If the OpenAI API call fails (e.g., quota), the app returns extracted snippets from the most relevant chunks so you still get useful context.
- To fully avoid usage, you can keep using the fallback mode results, or we can add a UI toggle on request.

## Configuration
- Auth users/passwords are in `config.yaml`.
- Vector stores are bundled as FAISS indexes in `faiss_pdf_1/` and `faiss_url_1/`.
- To rebuild indexes, see `loader.py` (requires valid OpenAI embedding access).

## Notes
- Never commit secrets. `.streamlit/secrets.toml` is gitignored.
- If you’ve previously committed a secret, rotate the key and rewrite history (e.g., via git-filter-repo).
