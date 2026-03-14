# ──────────────────────────────────────────────────────────
# chatbot.py — Legacy / reference file
# The active chatbot logic lives in main.py
# This file is kept for reference only.
# ──────────────────────────────────────────────────────────

import streamlit as st
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("GOOGLE_EMBED_MODEL", "models/gemini-embedding-001")


def get_vectorstore_from_url():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY
    )
    pdf_vector_store = FAISS.load_local(
        "./faiss_pdf_1", embeddings, allow_dangerous_deserialization=True
    )
    url_vector_store = FAISS.load_local(
        "./faiss_url_1", embeddings, allow_dangerous_deserialization=True
    )
    pdf_vector_store.merge_from(url_vector_store)
    return pdf_vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY
    )
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up "
         "in order to get information relevant to the conversation"),
    ])
    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL, google_api_key=GOOGLE_API_KEY
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return response["answer"]


def logout_button():
    st.query_params.key = "login"
    st.session_state["logout"] = True
    st.session_state["name"] = None
    st.session_state["username"] = None
    st.session_state["authentication_status"] = None
    st.session_state["app_page"] = None


def chatBot():
    try:
        st.session_state["app_page"] = True
        username = st.session_state["name"]
        username = username.capitalize()
        with st.sidebar:
            st.title(f"Hello, {username}")
            st.button("Logout", on_click=logout_button)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(
                    content=f"Hello, I am a Government Infobot. How can I help you {username}?"
                )
            ]
        if "vector_store" not in st.session_state:
            st.session_state.vector_store = get_vectorstore_from_url()

        user_query = st.chat_input("Type your message here...")
        if user_query is not None and user_query != "":
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)
    except Exception:
        st.rerun()
