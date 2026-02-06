"""
Digital Protection - Knowledge Base Ingestion Script
Supports both English and Arabic content (UTF-8)
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import shutil

def update_knowledge_base():
    """Load knowledge base files and create FAISS index"""
    
    print("--- Starting Knowledge Base Ingestion ---")
    
    # Path to data folder
    data_folder = "data"
    
    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"Error: '{data_folder}' folder not found!")
        print("Please create a 'data' folder and add your knowledge_base.txt file.")
        return
    
    # Find all text files
    all_documents = []
    
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_folder, filename)
            print(f"Loading: {file_path}")
            
            try:
                # Use UTF-8 encoding to support Arabic
                loader = TextLoader(file_path, encoding="utf-8")
                documents = loader.load()
                all_documents.extend(documents)
                print(f"  ✓ Loaded {len(documents)} document(s) from {filename}")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
                
                # Try alternative encoding if UTF-8 fails
                try:
                    print(f"  Trying alternative encoding...")
                    loader = TextLoader(file_path, encoding="utf-8-sig")
                    documents = loader.load()
                    all_documents.extend(documents)
                    print(f"  ✓ Loaded with utf-8-sig encoding")
                except Exception as e2:
                    print(f"  ✗ Still failed: {e2}")
                    continue
    
    if not all_documents:
        print("\nNo documents found! Please add .txt files to the 'data' folder.")
        return
    
    print(f"\n--- Total documents loaded: {len(all_documents)} ---")
    
    # Split documents into chunks
    print("\n--- Splitting documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Increased from 500
        chunk_overlap=150,     # Increased from 50 to keep more context between chunks
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\n--- Creating embeddings (this may take a moment) ---")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Create FAISS index
    print("\n--- Building FAISS vector store ---")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    index_path = "faiss_index"
    vectorstore.save_local(index_path)
    print(f"\n✓ FAISS index saved to '{index_path}/' folder")
    
    # Compress the index
    print("\n--- Compressing FAISS index ---")
    shutil.make_archive("faiss_index", 'zip', index_path)
    print("✓ Compressed to 'faiss_index.zip'")
    
    # Summary
    print("\n" + "="*50)
    print("SUCCESS! Knowledge base updated.")
    print("="*50)
    print(f"Documents processed: {len(all_documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Index saved to: {index_path}/")
    print("\nNext steps:")
    print("1. Upload the 'faiss_index' folder to GitHub")
    print("2. Reboot your Streamlit app")
    print("="*50)

if __name__ == "__main__":
    update_knowledge_base()