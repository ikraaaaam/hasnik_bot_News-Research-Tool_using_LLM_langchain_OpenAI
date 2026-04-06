from langchain.chains import RetrievalQA
from langchain import OpenAI

def setup_qa_retriever(vector_store, k=2):
    """
    Sets up the RetrievalQA chain using the provided vector store.
    """
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.9, max_tokens=500), 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    return qa
