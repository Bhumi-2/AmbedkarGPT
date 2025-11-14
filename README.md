# AmbedkarGPT - Intern Assignment (Kalpit Pvt Ltd, UK)

A simple **local** RAG prototype built exactly per the brief:

- **LangChain** for orchestration  
- **HuggingFaceEmbeddings**: `sentence-transformers/all-MiniLM-L6-v2`  
- **ChromaDB** as vector store (local, persisted on disk)  
- **Ollama (Mistral 7B)** as LLM (free, runs locally)  

> No API keys, no cloud calls. 

---

## Project Structure

```
AmbedkarGPT-Intern-Task/
├── main.py
├── requirements.txt
├── speech.txt
└── README.md
```

---

## Setup (Step-by-step)

### 1) Install Ollama & Pull Mistral
**macOS (Intel/M1/M2):**
```bash
brew install ollama || curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```

> Ensure the `ollama` service is running. On macOS, it starts automatically; on Linux you can run `ollama serve` in a terminal.

### 2) Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3) Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4) Run the App
```bash
python main.py
```
Now type a question, e.g.:
```
What is the real remedy according to the text?
```


---

## How it Works (RAG Pipeline)

1. **Load** `speech.txt`  
2. **Split** text into chunks (`RecursiveCharacterTextSplitter`)  
3. **Embed** chunks with `all-MiniLM-L6-v2` (HuggingFace)  
4. **Store** embeddings in **Chroma** (persisted in `./chroma_store`)  
5. **Retrieve** top-k relevant chunks for your query  
6. **Answer** using **Ollama (mistral)** with retrieved context  

---
