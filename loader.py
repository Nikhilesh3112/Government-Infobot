from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# Load API key from environment or .streamlit/secrets.toml
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        import tomllib
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
        with open(secrets_path, "rb") as f:
            secrets = tomllib.load(f)
        GOOGLE_API_KEY = secrets.get("google", {}).get("api_key", "")
    except Exception:
        pass

EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def scrape_and_store(urls, output_directory, flag):
    embeddings = get_embeddings()

    if flag == 1:
        # Web scraping mode
        for url in urls:
            if not url.startswith("https://") and not url.startswith("http://"):
                raise ValueError(f"Invalid URL: {url}. URL must start with 'https://' or 'http://'")

            loader = WebBaseLoader(url)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            document_chunks = text_splitter.split_documents(document)
            vector_store = FAISS.from_documents(document_chunks, embeddings)
            vector_store.save_local(output_directory)
            print(f"URL vector store saved to {output_directory}")
    else:
        # PDF directory mode
        loader = PyPDFDirectoryLoader(urls)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(document)
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        vector_store.save_local(output_directory)
        print(f"PDF vector store saved to {output_directory}")

if __name__ == "__main__":
    urls = ["https://en.wikipedia.org/wiki/List_of_schemes_of_the_government_of_India"]
    output_directory = "faiss_url_1"
    scrape_and_store(urls, output_directory, 1)

    # Uncomment to rebuild PDF index:
    # input_directory = os.path.join(os.getcwd(), "pdf")
    # scrape_and_store(input_directory, "faiss_pdf_1", 2)
