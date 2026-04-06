import os
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="Hasnik Bot", page_icon="🧠", layout="wide")

# --- UI Header ---
st.title("🧠 Hasnik Bot")
st.markdown("### *RAG-based News Research Assistant (LangChain + FAISS + OpenAI)*")

# --- Workflow Instructions ---
with st.expander("🚀 Quick Start Guide", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Step 1: Enter URLs**")
        st.caption("Add up to 3 news article URLs in the sidebar.")
    with col2:
        st.markdown("**Step 2: Process**")
        st.caption("Click 'Process URLs' to index the content.")
    with col3:
        st.markdown("**Step 3: Ask Questions**")
        st.caption("Get instant insights from the processed articles.")

st.divider()

# --- Sidebar: Input Section ---
st.sidebar.header("📁 Data Sources")
st.sidebar.markdown("Specify the news articles you want to analyze.")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}", placeholder="https://news-site.com/article")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs", type="primary")

# --- Main Logic: Data Processing ---
if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        with st.spinner("Step 1 of 2: Scraping and splitting content..."):
            data = ""
            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, "html.parser")
                    paragraphs = soup.find_all("p")
                    for paragraph in paragraphs:
                        data += paragraph.get_text().strip() + " "
                except Exception as e:
                    st.sidebar.error(f"Error fetching data: {url}")

            if data:
                # Maintain original character splitting logic
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_text(data)

                with st.spinner("Step 2 of 2: Building vector store with OpenAI..."):
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)
                    st.session_state.vectorestore_openai = vectorstore
                    st.success("✅ Analysis Ready: Documents processed and indexed successfully!")
            else:
                st.error("No content could be extracted from the provided URLs.")

# --- Main Area: Knowledge Query ---
st.header("🔍 Research Query")

# Check if vectorstore is initialized
if "vectorestore_openai" not in st.session_state:
    st.info("💡 **Waiting for data**: Please add URLs in the sidebar and click 'Process URLs' to enable research capability.")
else:
    query = st.text_input("What specific information are you looking for?", placeholder="e.g., What are the key takeaways from these articles?")

    if query:
        with st.spinner("Searching processed knowledge..."):
            try:
                llm = OpenAI(temperature=0.7, max_tokens=500)
                retriever = st.session_state.vectorestore_openai.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 3}
                )
                
                qa = RetrievalQA.from_chain_type(
                    llm=llm, 
                    chain_type="stuff", 
                    retriever=retriever, 
                    return_source_documents=True
                )

                result = qa({"query": query})
                
                # --- Improved Output Display ---
                st.subheader("📝 Insight")
                st.write(result['result'])
                
                # Source metadata (internal)
                with st.expander("View Retrieval Points", expanded=False):
                    st.caption("Retrieved relevant context from your processed URLs.")
                    # Optionally list documents here if metadata is improved later

            except Exception as e:
                st.error(f"An error occurred during retrieval: {e}")

# --- Footer ---
st.divider()
st.caption("Built with LangChain, OpenAI, and FAISS | Doc Oclock AI")