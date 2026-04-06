# <h1 align="center">Hasnik Bot – RAG-based News Research Assistant</h1>

<p align="center">
  <strong>Efficiently extract, index, and query insights from multiple news sources using Retrieval-Augmented Generation (RAG).</strong>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.0.284-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=for-the-badge&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00599C?style=for-the-badge&logo=facebook&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

</div>

---

## 🌐 Live Demo
> [!NOTE]
> *Deployment link coming soon! For now, follow the local installation steps below.*

---

## 📖 Overview
**Hasnik Bot** is a high-performance News Research Assistant built on a **Retrieval-Augmented Generation (RAG)** architecture. It empowers users to input multiple news URLs, process them into a searchable vector database, and perform semantic queries to extract precise information without reading through entire articles.

### ⚠️ Problem Statement
In the fast-paced financial and tech world, researchers often need to aggregate data from multiple news articles quickly. Manually scanning dozens of URLs is time-consuming and error-prone.

### ✅ Solution
Hasnik Bot automates the content extraction and knowledge retrieval process. By using **OpenAI Embeddings** and **FAISS**, it converts unstructured web data into structured, queryable knowledge, providing direct answers with source attribution.

---

## 🏗️ AI Architecture (RAG Pipeline)
The application follows a modular RAG pipeline to ensure accuracy and low latency:

1.  **Data Ingestion**: Scrapes content from provided URLs using `UnstructuredURLLoader`.
2.  **Preprocessing**: Cleans and splits the text into manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding Generation**: Converts text chunks into high-dimensional vectors using `OpenAIEmbeddings`.
4.  **Vector Storage**: Indexes vectors in **FAISS** (Facebook AI Similarity Search) for efficient similarity retrieval.
5.  **Retrieval & Generation**: When a user asks a question:
    -   The query is embedded.
    -   Top-$k$ relevant chunks are retrieved from FAISS.
    -   Retrieved context is passed to a **ChatOpenAI** LLM to generate a factual response with source citations.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **Multi-URL Loading** | Input up to 3 news URLs simultaneously for bulk analysis. |
| **Semantic Search** | Uses vector embeddings to understand the context of your questions. |
| **Contextual Answers** | Generates responses based *only* on the provided articles. |
| **Source Tracking** | Provides URLs of the articles used to generate the answer. |
| **Local Persistence** | Saves the FAISS index locally (pickle) to avoid re-processing same URLs. |

---

## 📸 Screenshots

### 🔍 Query & Results

![Query Result](docs/screenshots/image.PNG)

---

## 🛠️ Tech Stack

-   **Frontend**: [Streamlit](https://streamlit.io/) (Interactive Web Interface)
-   **Orchestration**: [LangChain](https://www.langchain.com/) (LLM Framework)
-   **LLM**: OpenAI GPT-3.5-turbo
-   **Embeddings**: OpenAI `text-embedding-ada-002`
-   **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss)
-   **Parsing**: [Unstructured](https://unstructured.io/)

---

## 📂 Project Structure
```text
.
├── app/
│   └── main.py             # Streamlit entry point & UI logic
├── rag/
│   ├── embeddings.py       # OpenAI embedding configuration
│   ├── pipeline.py         # RetrievalQA chain logic
│   └── retriever.py        # FAISS index management
├── utils/
│   ├── scraper.py          # URL content extraction
│   └── text_splitter.py    # Recursive character splitting logic
├── docs/                   # Documentation & Screenshots
├── notebooks/              # Prototyping and R&D
├── .env                    # System environment variables
├── requirements.txt        # Production dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ikraaaaam/hasnik_bot_News-Research-Tool_using_LLM_langchain_OpenAI.git
   cd hasnik_bot_News-Research-Tool_using_LLM_langchain_OpenAI
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔑 Environment Setup
Create a `.env` file in the root directory and add your OpenAI API Key:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

---

## 🚀 Usage

1. **Start the Application**
   ```bash
   streamlit run app/main.py
   ```

2. **Analysis Flow**
   - Enter news article URLs in the sidebar.
   - Click **"Process URLs"** to trigger the RAG pipeline.
   - Once processing completes, type your research question in the main chat input.

---

## 📈 Future Improvements
- [ ] Integration with open-source LLMs (Llama 3/Mistral) via Ollama.
- [ ] Add support for PDF and local text file ingestion.
- [ ] Implement Pinecone/Weaviate for cloud vector storage.
- [ ] Enhanced UI with chat history and exportable PDF summaries.

---

## 👨‍💻 Author
**Ikram**  
*AI Engineer & Research Enthusiast*

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ikraaaaam)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/md-ekram-ullah26/)

---

<p align="center">
  Built with ❤️ for AI Researchers
</p>
