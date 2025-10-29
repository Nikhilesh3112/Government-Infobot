import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import time
import os
from streamlit_authenticator.utilities.exceptions import (CredentialsError,
  ForgotError,
  LoginError,
  RegisterError,
  ResetError,
  UpdateError)
# from chatbot import chatBot


# import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader, OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# st.markdown(
#     """
#   <style>
#       section[data-testid="stSidebar"] {
#           width: 10px !important; # Set the width to your desired value
#       }
#       .stApp{
#         width: 1000px;
#         height: 800px;
#       }
#   </style>
#   """,unsafe_allow_html=True,
# )
st.set_page_config(page_title="Government Infobot", page_icon="🤖")
st.title("Government Infobot")

def get_vectorstore_from_url():
  # output_directory=os.getcwd()
  # print(output_directory)
  # output_directory_pdf=output_directory+"/chroma_db_pdf"
  # files=os.listdir(output_directory_pdf)
  # # print(files)
  # db_file=files[1]
  # print(db_file)
  # # def get_vectorstore_from_disk(output_directory, vector_store_index):
  # vector_store_file = os.path.join(output_directory, db_file)
  # vector_store = Chroma(persist_directory=vector_store_file,embedding_function=OpenAIEmbeddings())  # Corrected line
  # output_directory_url=output_directory+"/chroma_db"
  # files=os.listdir(output_directory_url)
  # db_file=files[1]
  # print(db_file)
  # vector_store_file = os.path.join(output_directory, db_file)
  # vector_store = Chroma(persist_directory=vector_store_file,embedding_function=OpenAIEmbeddings())
  pdf_vector_store = FAISS.load_local("./faiss_pdf_1",
                                      OpenAIEmbeddings(model=EMBED_MODEL),
                                      allow_dangerous_deserialization=True)
  url_vector_store = FAISS.load_local("./faiss_url_1",
                                      OpenAIEmbeddings(model=EMBED_MODEL),
                                      allow_dangerous_deserialization=True)
  pdf_vector_store.merge_from(url_vector_store)
  vector_store = pdf_vector_store
  # vector_store=vector_store.append(url_vector_store)
  # url_vector_store=url_vector_store.append(pdf_vector_store)
  # vector_store= combine_vector_stores(url_vector_store,pdf_vector_store)
  return vector_store


def _format_context(docs):
  try:
    texts = [getattr(d, 'page_content', str(d)) for d in docs]
  except Exception:
    texts = [str(d) for d in docs]
  # Limit total context to ~2000 characters for free-tier usage
  joined = "\n\n".join(texts)
  return joined[:2000]


def _retrieve(vector_store, query):
  # Reduce number of retrieved chunks to control cost/latency
  retriever = vector_store.as_retriever(search_kwargs={"k": 2})
  return retriever.invoke(query)


def get_response(user_input):
  docs = _retrieve(st.session_state.vector_store, user_input)
  context = _format_context(docs)
  llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2, max_tokens=300, max_retries=0, timeout=20)
  messages = []
  messages.append(AIMessage(content=f"You are a helpful assistant. Answer the user's questions based only on the provided context. If the answer is not in the context, say you don't know.\n\nContext:\n{context}"))
  messages.extend(st.session_state.chat_history)
  messages.append(HumanMessage(content=user_input))
  try:
    resp = llm.invoke(messages)
    return resp.content
  except Exception as e:
    # Fallback: extractive answer from retrieved docs when LLM is unavailable (e.g., quota exceeded)
    snippet = context[:1200]
    return (
      "I'm temporarily unable to generate a full answer due to API limits. "
      "Here's the most relevant information I found:\n\n" + snippet +
      "\n\nTip: add billing to your OpenAI account or use a different API key to enable full answers."
    )


# # app config
# def logout_button():
#   st.query_params.key = "login"
#   st.session_state['logout'] = True
#   st.session_state['name'] = None
#   st.session_state['username'] = None
#   st.session_state['authentication_status'] = None
  # st.session_state['app_page'] = None


def chatBot():
  try:
    # st.session_state['app_page'] = True
    username = st.session_state.get('name') or st.session_state.get('username') or "User"
    # username = "nithish"
    username = str(username).capitalize()
    with st.sidebar:
      st.title(f"Hello, {username}")
      st.button("Logout", on_click=logout_button)
    # Ensure initial greeting is present
    if "chat_history" not in st.session_state:
      st.session_state.chat_history = []
      st.session_state.chat_history.append(
          AIMessage(
              content=
              f"Hello, I am a Government Infobot. How can I help you {username}?"
          )
      )
    # Load vector store guarded to avoid blank screen during long load/errors
    if "vector_store" not in st.session_state:
      try:
        with st.spinner("Loading knowledge base..."):
          st.session_state.vector_store = get_vectorstore_from_url()
      except Exception as e:
        st.error(f"Failed to load knowledge base: {e}")
        st.info("You can still type a question; I'll try to respond without retrieval.")
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
      response = get_response(user_query)
      st.session_state.chat_history.append(HumanMessage(content=user_query))
      st.session_state.chat_history.append(AIMessage(content=response))

      # conversation
    for message in st.session_state.chat_history:
      if isinstance(message, AIMessage):
        with st.chat_message("AI"):
          st.write(message.content)
      elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
          st.write(message.content)
  except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Debug info:")
    st.write(f"Authentication status: {st.session_state.get('authentication_status')}")
    st.write(f"Name: {st.session_state.get('name')}")
    st.write(f"Username: {st.session_state.get('username')}")


# Check if set_page_config has been executed already
if not st.session_state.get("has_set_page_config", False):
    # Execute set_page_config
    
    # Set the flag to indicate that set_page_config has been executed
    st.session_state["has_set_page_config"] = True
# st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
#                       'Get Help': 'https://www.extremelycoolapp.com/help',
#                       'Report a bug': "https://www.extremelycoolapp.com/bug",
#                       'About': "# This is a header. This is an *extremely* cool app!"
#                   })
# st.title("Government Infobot")


def saveToYaml():
    with open('./config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)
# Loading config file
with open('./config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.load(file, Loader=SafeLoader)


def apply_openai_settings():
    global OPENAI_API_KEY, CHAT_MODEL, EMBED_MODEL
    api_key = None
    chat_model = None
    embed_model = None
    # 1) Try Streamlit secrets
    try:
        secrets_openai = st.secrets.get("openai", {})
        api_key = secrets_openai.get("api_key") or api_key
        chat_model = secrets_openai.get("chat_model") or chat_model
        embed_model = secrets_openai.get("embed_model") or embed_model
    except Exception:
        pass
    # 2) Try config.yaml openai section
    cfg_openai = (config or {}).get("openai", {}) if isinstance(config, dict) else {}
    api_key = cfg_openai.get("api_key") or api_key
    chat_model = cfg_openai.get("chat_model") or chat_model
    embed_model = cfg_openai.get("embed_model") or embed_model
    # 3) Fallback to existing env/defaults already set in globals
    # Apply to environment for downstream libs
    if api_key and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
        OPENAI_API_KEY = api_key
    if chat_model:
        os.environ["OPENAI_CHAT_MODEL"] = chat_model
        CHAT_MODEL = chat_model
    if embed_model:
        os.environ["OPENAI_EMBED_MODEL"] = embed_model
        EMBED_MODEL = embed_model


# Apply OpenAI settings early
apply_openai_settings()


# Creating the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# st.markdown(
#     """
#     <style>
#         section[data-testid="stSidebar"] {
#             width: 80px !important; # Set the width to your desired value
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
st.markdown(
    """
  <title>Chat with websites</title>
  <style>
      html {
        font-size: 20px;
      }
      section[data-testid="stSidebar"] {
          width: 10px !important; # Set the width to your desired value
      }
      /*.st-emotion-cache-1eo1tir {
        padding: 1rem 1rem 1rem 0rem;
      }
      .st-emotion-cache-arzcut {
        padding: 1rem 1rem 1rem 0rem;
      }*/
  </style>
  """,unsafe_allow_html=True,
)
def logout_button():
    st.query_params.key="login"
    st.session_state['logout'] = True
    st.session_state['name'] = None
    st.session_state['username'] = None
    st.session_state['authentication_status'] = None
    if 'chat_history' in st.session_state:
        del st.session_state['chat_history']
    if 'vector_store' in st.session_state:
        del st.session_state['vector_store']

    # st.rerun()

def app():
    # st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
    #                       'Get Help': 'https://www.extremelycoolapp.com/help',
    #                       'Report a bug': "https://www.extremelycoolapp.com/bug",
    #                       'About': "# This is a header. This is an *extremely* cool app!"
    #                   })
    # st.title("Government Infobot")
    # st.write("Welcome to the Government Infobot")
    # st.rerun()
    chatBot()

def login():
    try:
        # print(st.session_state['authentication_status'])
        login_result = authenticator.login()
        if isinstance(login_result, tuple) and len(login_result) == 3:
            (name_of_the_user, login_status, username) = login_result
        # print("Login Status",login_status)
        if st.session_state['authentication_status']:
            st.success("Logged in successfully!")
            # time.sleep(5)
            st.query_params.key="app"
            st.rerun()
    except LoginError as e:
        st.error(e)
    st.markdown("Don't have an account? [Register here](/?key=register)")

def register():
    try:
        preauth_list = config.get('preauthorized', {}).get('emails', [])
        (email_of_registered_user,username_of_registered_user,name_of_registered_user) = authenticator.register_user(pre_authorization=preauth_list)
        if email_of_registered_user:
            st.success('User registered successfully')
            saveToYaml()
            st.query_params.key="login"
            st.rerun()

    except RegisterError as e:
        st.error(e)
    st.markdown("Done Registering? [Login here](/?key=login)")


def main():
    # Only reset session on cold start (no key) or when explicitly on login AND not authenticated
    if "key" not in st.query_params:
        st.query_params.key = "login"
        st.session_state['logout'] = True
        st.session_state['name'] = None
        st.session_state['username'] = None
        st.session_state['authentication_status'] = None
        # st.session_state['app_page'] = None
    elif st.query_params.key == "login" and st.session_state.get('authentication_status') is None:
        st.session_state['logout'] = True
        st.session_state['name'] = None
        st.session_state['username'] = None
        st.session_state['authentication_status'] = None
        # st.session_state['app_page'] = None
    page = st.query_params["key"]
    print(page)
    # If already authenticated, always route to app
    if st.session_state.get('authentication_status') is True and page != "app":
        st.query_params.key = "app"
        st.rerun()
    if page == "login" and st.session_state['authentication_status'] == None:
        login()
    elif page == "register":
        register()
    elif page == "app":
        app()


if __name__ == "__main__":
    # st.set_page_config(page_title="Chat with websites", page_icon="🤖",menu_items={
    #                                         'Get Help': 'https://www.extremelycoolapp.com/help',
    #                                         'Report a bug': "https://www.extremelycoolapp.com/bug",
    #                                         'About': "# This is a header. This is an *extremely* cool app!"
    #                                     })
    # st.title("Government Infobot")
    main()
