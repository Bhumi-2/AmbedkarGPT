# AmbedkarGPT 

A simple **local Retrieval-Augmented Generation (RAG)** system built exactly according to the assignment requirements.

This project uses:

- **Python 3.8+**
- **LangChain** for RAG orchestration  
- **HuggingFace Embeddings** (`sentence-transformers/all-MiniLM-L6-v2`)  
- **ChromaDB** as a local vector store  
- **Ollama** with **Mistral 7B** as the local LLM  

> 100% local. No API keys. No cloud usage.  

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

# Setup Instructions (macOS / Linux)

Follow these steps to run the project locally.

---

## **1) Install Ollama & Pull the Mistral 7B Model**

### **macOS (Recommended: Homebrew)**
```bash
brew install ollama
brew services start ollama
```

Or download the macOS installer:

 https://ollama.com/download

Then pull the required model:
```bash
ollama pull mistral
```

### **Linux**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral
```

Verify Ollama works:
```bash
ollama run mistral "Say OK"
```

---

## **2) Create & Activate a Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
# Windows: venv\Scripts\activate
```

---

## **3) Install Python Dependencies**

```bash
pip install -r requirements.txt
```

---

## **4) Run the Application**

```bash
python3 main.py
```

Ask any question based on the `speech.txt`, for example:

```
What is the real remedy according to the text?
```

To exit:
```
exit
```

---

# How the RAG System Works

1. **Load** `speech.txt`  
2. **Split** it into smaller text chunks (`RecursiveCharacterTextSplitter`)  
3. **Embed** chunks using HuggingFace “all-MiniLM-L6-v2”  
4. **Store** embeddings in a **local ChromaDB**  
5. **Retrieve** the most relevant chunks for the query  
6. **Feed** retrieved chunks + user query → **Ollama (Mistral 7B)**  
7. **Generate** the final answer  

---

# Example Output

**Input:**

```
What is the real remedy according to the text?
```

**Output:**

```
The real remedy, according to the given context, is to destroy the belief in the sanctity of the shastras.
```

---

# Notes

- The vector store is saved in `./chroma_store` for faster reuse.
- The system works fully offline once the model is downloaded.
- No API keys, accounts, or cloud services are required.

