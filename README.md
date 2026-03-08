# Digital Protection RAG Chatbot

A production-ready bilingual (English/Arabic) AI chatbot built with RAG (Retrieval-Augmented Generation) for a cybersecurity consultancy.


##  Live Demo

[View Live Chatbot] ((https://dp-technologies.net))

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    User     │ →   │  Streamlit  │ →   │    FAISS    │ →   │  Groq API   │
│   Query     │     │   Cloud     │     │  Vector DB  │     │ Llama 70B   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ↓
                                                            ┌─────────────┐
                                                            │  Response   │
                                                            │  to User    │
                                                            └─────────────┘
```

##  Features

- **Bilingual Support:** English and Arabic with proper RTL alignment
- **RAG Pipeline:** Retrieves relevant context from knowledge base before answering
- **Fast Inference:** Uses Groq API for sub-second response times
- **Embedded Widget:** Can be embedded in any website via iframe
- **Mobile Responsive:** Works on all devices
- **Guardrails:** No emojis, no legal advice, no specific pricing

##  Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Llama 3.3 70B (via Groq) |
| **Embeddings** | paraphrase-multilingual-MiniLM-L12-v2 & llama-3.3-70b-versatile|
| **Vector Database** | FAISS |
| **Framework** | LangChain |
| **Frontend** | Streamlit |
| **Deployment** | Streamlit Cloud |

##  Project Structure

```
├── app.py                 # Main Streamlit application
├── ingest_data.py         # Script to create FAISS index
├── requirements.txt       # Python dependencies
├── faiss_index/           # Vector store
│   ├── index.faiss
│   └── index.pkl
└── .streamlit/
    └── config.toml        # Streamlit configuration (confidential) 
```

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dp-chatbot.git
cd dp-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Ingest the knowledge base
```bash
python ingest_data.py
```

### 5. Run the app
```bash
streamlit run app.py
```

## Configuration

### Changing the Knowledge Base
1. Edit `data/knowledge_base.txt` with your content
2. Run `python ingest_data.py` to rebuild the FAISS index
3. Restart the Streamlit app

### Adjusting RAG Parameters
In `app.py`, you can modify:
```python
# Number of chunks to retrieve (default: 4)
return vectorstore.as_retriever(search_kwargs={"k": 4})

# Model selection
GROQ_MODEL = "llama-3.3-70b-versatile"
BACKUP_MODEL = "llama-3.1-8b-instant"
```

## Key Learnings

1. **Model Size:** Started with 1B, moved to 70B for reliable instruction following
2. **Chunk Size:** k=4 provides the best balance between context and accuracy
3. **Embeddings:** Multilingual embeddings are essential for non-English support
4. **Groq:** Provides fast, free inference for Llama models

## Embedding in a Website

Use the chat widget HTML to embed on any website:
```html
<!-- Add before </body> -->
<script>
  // See chat-widget.html for full implementation
</script>
```

## License

MIT License - feel free to use this for your own projects!

## Contributing

Contributions welcome! Please open an issue or PR.

##  Contact

- **Email:** iyas.shraim@gmail.com
- **Website:** [dp-technologies.net](https://dp-technologies.net)

---

Built with using LangChain, Groq, and Streamlit
