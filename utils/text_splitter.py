from langchain.text_splitter import CharacterTextSplitter

def split_text_to_docs(data, chunk_size=100, chunk_overlap=20):
    """
    Splits a large string of data into manageable documents using CharacterTextSplitter.
    """
    if not data:
        return []
    
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_text(data)
    return docs
