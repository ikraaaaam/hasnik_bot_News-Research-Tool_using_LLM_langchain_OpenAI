import os
import streamlit as st
import pickle
import time
import requests
from bs4 import BeautifulSoup
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import WeaviateVectorStore  # Import WeaviateVectorStore
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit UI
st.title("Hasnik Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect user input for URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

# Check if the "Process URLs" button is clicked
process_url_clicked = st.sidebar.button("Process URLs")

# Define the file path for storing the Weaviate vector store
file_path = "weaviate_store.pkl"

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
                data += paragraph.get_text() + "\n\n"
        except Exception as e:
            st.error(f"Error fetching data from {url}: {e}")

    print(type(data))

    # Split the data into documents using the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.create_documents(data)
    print(docs)

    # Create a Weaviate vector store with specified attributes
    vectorstore_weaviate = WeaviateVectorStore(weaviate_client, "NewsDocuments", attributes=["source"])

    # Add documents to the Weaviate vector store
    vectorstore_weaviate.add_documents(docs)
    main_placeholder.text("Vector Store...Started...✅✅✅")
    time.sleep(2)

    # Save the Weaviate vector store to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_weaviate, f)

    print('done')

# Collect user input for the question
query = st.text_input("Question: ")

# Process the question and retrieve an answer if the question is provided
if query:
    try:
        # Load the Weaviate vector store from the pickle file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Use LangChain to find an answer to the question
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])
    except Exception as e:
        st.error(f"An error occurred while answering the question: {e}")
