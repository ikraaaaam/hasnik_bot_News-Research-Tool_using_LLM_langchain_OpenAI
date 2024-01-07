import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
#from langchain_community.document_loaders import BSHTMLLoader

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit UI
st.title("Hasnik Bot: URL Analyzer! 📈")
st.sidebar.title("Put URLs")

# Collect user input for URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

# Check if the "Process URLs" button is clicked
process_url_clicked = st.sidebar.button("Process URLs")

# Define the file path for storing the FAISS index
#file_path = "faiss_store.pkl"

# Create a placeholder for displaying messages
main_placeholder = st.empty()

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.9, max_tokens=500)



# Process URLs if the button is clicked
if process_url_clicked:

    data = ""
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            for paragraph in paragraphs:
                # Add the source metadata variable for each paragraph
                data += paragraph.get_text().strip()
        except Exception as e:
            st.error(f"Error fetching data from {url}: {e}")

    

#Split the data into documents using the RecursiveCharacterTextSplitter
    text_splitter = CharacterTextSplitter( chunk_size=100,chunk_overlap=20)
    docs= text_splitter.split_text(data)                       
    


 # Create embeddings and save them to the FAISS index
    embeddings = OpenAIEmbeddings()
    st.session_state.vectorestore_openai =FAISS.from_texts(texts= docs, embedding= embeddings)
    main_placeholder.text("Embedding Vector Building...✅✅✅")
    time.sleep(2)


# Collect user input for the question
query = st.text_input("Question: ")


# Process the question and retrieve an answer if the question is provided
if query:
    try:
       
        retriever =   st.session_state.vectorestore_openai.as_retriever(search_type="similarity", search_kwargs={"k":2})
        
        qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

        

        result =  qa({"query": query})
        
        answer = result['result']
        
        st.header("Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"An error occurred while answering the question: {e}")
