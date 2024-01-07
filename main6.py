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
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Set up Streamlit UI
st.title("Hasnik Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Collect user input for URLs
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]

# Check if the "Process URLs" button is clicked
process_url_clicked = st.sidebar.button("Process URLs")

# Define the file path for storing the FAISS index
file_path = "faiss_store_openai.pkl"

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


 # Create embeddings and save them to the FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

    print('done')

# Collect user input for the question
query = st.text_input("Question: ")

# Process the question and retrieve an answer if the question is provided
if query:
    try:
        # Load the FAISS index from the pickle file
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
        qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

        response = qa({"question": query}, return_only_outputs=True)
        answer = response['answer']
        st.write(answer)

        # Use LangChain to find an answer to the question
        #chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        #result = qa ({"question": query}, return_only_outputs=True)
        #result= qa.run(query=query)
        #answer = result["result"]
        #source_documents = result["source_documents"]


        # Display the answer
        #st.header("Answer")
        #st.write(result["answer"])
    except Exception as e:
        st.error(f"An error occurred while answering the question: {e}")
