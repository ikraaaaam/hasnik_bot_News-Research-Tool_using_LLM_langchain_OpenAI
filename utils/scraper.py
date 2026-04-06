import requests
from bs4 import BeautifulSoup
import streamlit as st

def scrape_urls(urls):
    """
    Scrapes content from a list of URLs and concatenates the text from <p> tags.
    """
    data = ""
    for url in urls:
        if not url.strip():
            continue
        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes
            soup = BeautifulSoup(response.content, "html.parser")
            paragraphs = soup.find_all("p")
            for paragraph in paragraphs:
                data += paragraph.get_text().strip() + " "
        except Exception as e:
            st.error(f"Error fetching data from {url}: {e}")
    return data
