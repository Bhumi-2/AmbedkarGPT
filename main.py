# main.py
import argparse
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


PERSIST_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_store(file_path: str, persist_dir: str) -> Chroma:
    """Load text, split into chunks, embed, and persist in Chroma."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please keep speech.txt beside main.py.")
    print("Loading document...")
    loader = TextLoader(file_path)
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs = splitter.split_documents(documents)

    print("Creating embeddings (HuggingFace all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Building Chroma vector store (this may take a moment)...")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f"Vector store created at: {persist_dir}")
    return vectordb

def load_or_build_vector_store(file_path: str, persist_dir: str) -> Chroma:
    """Try to load an existing persisted Chroma DB; build if missing."""
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        print("Using existing Chroma vector store.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        print("⚙️ No existing store found. Creating a new one...")
        return build_vector_store(file_path, persist_dir)

def create_qa_chain(vectordb: Chroma) -> RetrievalQA:
    """Create a RetrievalQA chain that uses the vector store as retriever and Ollama (Mistral) as LLM."""
    print("Initializing local LLM (Ollama • mistral)...")
    llm = Ollama(model="mistral")
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def interactive_loop(qa: RetrievalQA):
    """Simple REPL for asking questions."""
    print("\nAmbedkarGPT - Ask questions about 'speech.txt'. Type 'exit' to quit.")
    while True:
        q = input("\n Your question: ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not q:
            continue
        ans = qa.run(q)
        print(f"\n Answer: {ans}")

def main():
    parser = argparse.ArgumentParser(description="AmbedkarGPT Q&A over local speech.txt using LangChain + Chroma + Ollama")
    parser.add_argument("--rebuild", action="store_true", help="Force-rebuild the Chroma store.")
    parser.add_argument("--ask", type=str, default=None, help="Ask a single question non-interactively and exit.")
    parser.add_argument("--file", type=str, default="speech.txt", help="Path to the speech.txt file.")
    parser.add_argument("--store", type=str, default=PERSIST_DIR, help="Chroma persist directory.")
    args = parser.parse_args()

    # Rebuild vector store if requested
    if args.rebuild and os.path.isdir(args.store):
        # remove files in the persist dir
        for root, dirs, files in os.walk(args.store, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(args.store)
        print("🧹 Removed existing vector store. Rebuilding...")

    vectordb = load_or_build_vector_store(args.file, args.store)
    qa = create_qa_chain(vectordb)

    if args.ask:
        ans = qa.run(args.ask)
        print(f"\n Answer: {ans}")
    else:
        interactive_loop(qa)

if __name__ == "__main__":
    main()
