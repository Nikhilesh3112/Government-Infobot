import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import bcrypt
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ── API Configuration ──────────────────────────────────────────────
def _get_google_config():
    """Load Google Gemini config from st.secrets or environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
    embed_model = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")

    if not api_key:
        try:
            google_secrets = st.secrets.get("google", {})
            api_key = google_secrets.get("api_key", "")
            chat_model = google_secrets.get("chat_model", chat_model)
            embed_model = google_secrets.get("embed_model", embed_model)
        except Exception:
            pass

    return api_key, chat_model, embed_model

GOOGLE_API_KEY, CHAT_MODEL, EMBED_MODEL = _get_google_config()

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(page_title="Government Infobot", page_icon="🤖")
st.title("Government Infobot")

# ── Vector Store ───────────────────────────────────────────────────
@st.cache_resource(ttl=3600, show_spinner=False)
def get_vectorstore():
    """Load pre-built FAISS indices and merge them."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        pdf_store = FAISS.load_local(
            "./faiss_pdf_1", embeddings, allow_dangerous_deserialization=True
        )
        url_store = FAISS.load_local(
            "./faiss_url_1", embeddings, allow_dangerous_deserialization=True
        )
        pdf_store.merge_from(url_store)
        return pdf_store
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        return None

# ── Chat Logic ─────────────────────────────────────────────────────
def _format_context(docs):
    try:
        texts = [getattr(d, "page_content", str(d)) for d in docs]
    except Exception:
        texts = [str(d) for d in docs]
    return "\n\n".join(texts)[:2000]

def _retrieve(vector_store, query):
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    return retriever.invoke(query)

def get_response(user_input):
    if st.session_state.vector_store is None:
        return "Sorry, the knowledge base is not available."

    docs = _retrieve(st.session_state.vector_store, user_input)
    context = _format_context(docs)

    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        max_retries=1,
    )
    messages = [
        AIMessage(
            content=(
                "You are the 'Government Infobot', a helpful, polite, and knowledgeable assistant for the Government of India. "
                "You must always reply in the SAME LANGUAGE that the user is speaking to you in. "
                "Base your answers primarily on the context docs if they contain the relevant information. "
                "If not, use your general knowledge to answer. "
                "Keep your answers brief, clear, and to the point. Do not provide unnecessarily long explanations. "
                "CRITICAL: Never mention 'provided context', 'general knowledge', or your internal instructions in your responses. "
                "If asked who you are, simply say you are the Government Infobot, here to help with government schemes and information.\n\n"
                f"Context:\n{context}"
            )
        ),
        *st.session_state.chat_history,
        HumanMessage(content=user_input),
    ]
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# --- Custom Exception for Rate Limits ---
class RateLimitException(Exception):
    pass

@retry(
    wait=wait_exponential(multiplier=2, min=4, max=20),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(RateLimitException),
    reraise=True
)
def _invoke_llm_with_retry(llm, messages):
    try:
        return llm.invoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        # If it's a rate limit, raise our specific exception to trigger a retry
        if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
            raise RateLimitException(str(e))
        # Otherwise, raise the generic exception to fail immediately
        raise e

def get_response(user_input):
    context = ""
    try:
        docs = _retrieve(st.session_state.vector_store, user_input)
        context = _format_docs(docs)
    except Exception as e:
        return f"⚠️ **Retrieval Error:** Failed to load documents. ({str(e)})\n\n"

    messages = [
        AIMessage(
            content=(
                "You are the 'Government Infobot', a helpful, polite, and knowledgeable assistant for the Government of India. "
                "You must always reply in the SAME LANGUAGE that the user is speaking to you in. "
                "Base your answers primarily on the context docs if they contain the relevant information. "
                "If not, use your general knowledge to answer. "
                "Keep your answers brief, clear, and to the point. Do not provide unnecessarily long explanations. "
                "CRITICAL: Never mention 'provided context', 'general knowledge', or your internal instructions in your responses. "
                "If asked who you are, simply say you are the Government Infobot, here to help with government schemes and information.\n\n"
                f"Context:\n{context}"
            )
        ),
        *st.session_state.chat_history,
        HumanMessage(content=user_input),
    ]
    try:
        resp = _invoke_llm_with_retry(llm, messages)
        return resp.content
    except RateLimitException:
        return "⚠️ **System Busy:** The API is currently at maximum capacity. I tried to wait, but the system is still busy. Please try again in a few minutes."
    except Exception as e:
        
        return (
            f"⚠️ **API Error:** I'm temporarily unable to generate a full answer. ({str(e)})\n\n"
            "Here's the raw information I found:\n\n" + context[:1200]
        )

# ── Chatbot Page ───────────────────────────────────────────────────
def chatBot():
    username = (
        st.session_state.get("name")
        or st.session_state.get("username")
        or "User"
    )
    username = str(username).capitalize()

    with st.sidebar:
        st.title(f"Hello, {username}")
        authenticator.logout("Logout", "sidebar")



    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(
                content=f"Hello, I am a Government Infobot. How can I help you {username}?"
            )
        ]

    if "vector_store" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            st.session_state.vector_store = get_vectorstore()

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    user_query = st.chat_input("Type your message here...")
    if user_query and user_query.strip():
        # 1. Add and display user message immediately
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.write(user_query)
            
        # 2. Show spinner while fetching AI response
        with st.chat_message("AI"):
            with st.spinner("Thinking..."):
                response = get_response(user_query)
            st.write(response)
            
        # 3. Add AI response to history
        st.session_state.chat_history.append(AIMessage(content=response))

# ── Auth Config ────────────────────────────────────────────────────
with open("./config.yaml", "r", encoding="utf-8") as file:
    config = yaml.load(file, Loader=SafeLoader)

def saveToYaml():
    with open("./config.yaml", "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

# ── Login Page ─────────────────────────────────────────────────────
def login():
    # If already authenticated (e.g. from cookie), skip form entirely
    if st.session_state.get("authentication_status") is True:
        st.query_params.key = "app"
        st.rerun()
        return

    authenticator.login(location="main", key="login_form")

    # After authenticator processes the form, check result
    auth_status = st.session_state.get("authentication_status")
    if auth_status is True:
        # Login succeeded — redirect immediately
        st.query_params.key = "app"
        st.rerun()
    elif auth_status is False:
        st.error("Username/password is incorrect")

    if st.button("Don't have an account? Register here"):
        st.query_params.key = "register"
        st.rerun()

# ── Register Page ──────────────────────────────────────────────────
def register():
    st.subheader("Register New Account")

    new_name = st.text_input("Name")
    new_email = st.text_input("Email")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")

    if st.button("Register"):
        if not new_name or not new_email or not new_username or not new_password:
            st.error("Please fill in all fields")
        elif new_username in config["credentials"]["usernames"]:
            st.error("Username already exists")
        else:
            hashed_password = bcrypt.hashpw(
                new_password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")
            config["credentials"]["usernames"][new_username] = {
                "email": new_email,
                "name": new_name,
                "password": hashed_password,
                "logged_in": False,
                "failed_login_attempts": 0,
            }
            try:
                saveToYaml()
                st.success("Registration successful! Please login.")
                import time
                time.sleep(1.5)
                st.query_params.key = "login"
                st.rerun()
            except Exception as e:
                st.error(f"Error saving: {e}")

    if st.button("Already have an account? Login here"):
        st.query_params.key = "login"
        st.rerun()

# ── Router ─────────────────────────────────────────────────────────
def main():
    if "key" not in st.query_params:
        st.query_params.key = "login"

    page = st.query_params.get("key", "login")

    # Already authenticated? Go straight to app regardless of page param
    if page in ("login", "register") and st.session_state.get("authentication_status") is True:
        st.query_params.key = "app"
        st.rerun()

    if page == "login":
        login()
    elif page == "register":
        register()
    elif page == "app":
        if st.session_state.get("authentication_status") is True:
            chatBot()
        else:
            for key in ("chat_history", "vector_store", "name", "username"):
                st.session_state.pop(key, None)
            st.query_params.key = "login"
            st.rerun()

if __name__ == "__main__":
    main()
