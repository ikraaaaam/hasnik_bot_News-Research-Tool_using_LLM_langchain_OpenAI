from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st

def create_and_store_embeddings(docs):
    """
    Creates embeddings for the provided documents and stores the FAISS index in the Streamlit session state.
    """
    if not docs:
        st.warning("No documents to embed.")
        return False
        
    try:
        embeddings = OpenAIEmbeddings()
        # Storing in session_state as per the original architecture
        st.session_state.vectorestore_openai = FAISS.from_texts(texts=docs, embedding=embeddings)
        return True
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return False

def get_vector_store():
    """
    Retrieves the FAISS vector store from the session state.
    """
    return st.session_state.get("vectorestore_openai")
