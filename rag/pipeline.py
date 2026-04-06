import time
import streamlit as st
from utils.scraper import scrape_urls
from utils.text_splitter import split_text_to_docs
from rag.embeddings import create_and_store_embeddings
from rag.retriever import setup_qa_retriever

def ingest_data(urls, status_placeholder):
    """
    End-to-end pipeline to scrape, split, and embed documents.
    """
    status_placeholder.text("Fetching and scraping URLs... 🔄")
    data = scrape_urls(urls)
    
    if not data:
        status_placeholder.warning("No valid data found in URLs.")
        return False
        
    status_placeholder.text("Splitting text... 🔄")
    docs = split_text_to_docs(data)
    
    status_placeholder.text("Building Embedding Vector... 🔄")
    success = create_and_store_embeddings(docs)
    
    if success:
        status_placeholder.success("Embedding Vector Building...✅✅✅")
        time.sleep(2)
        status_placeholder.empty()
        return True
    return False

def answer_query(query, vector_store):
    """
    Processes the query, retrieves documents, and returns the answer.
    """
    try:
        qa = setup_qa_retriever(vector_store)
        result = qa({"query": query})
        return result['result']
    except Exception as e:
        st.error(f"An error occurred while answering the question: {e}")
        return None
