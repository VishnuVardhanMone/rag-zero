# RAG.zero

A browser-based Retrieval-Augmented Generation (RAG) system that lets you chat with any document — built with React, TF-IDF embeddings, and cosine similarity.

## What it does

- Upload any PDF or paste any text
- Asks questions about your document in natural language
- Finds the most relevant parts of your document using TF-IDF + cosine similarity
- Sends only the relevant context to the AI — not the whole document
- Powered by LLaMA 3.1 via Groq (free, fast)

## How RAG works
```
Your Document
      ↓
  Chunking (split into 300-word pieces)
      ↓
  TF-IDF Vectorization (convert text to math)
      ↓
  Cosine Similarity (find chunks closest to your question)
      ↓
  Top 3 chunks sent to LLaMA as context
      ↓
  Answer grounded in your document
```

## Tech Stack

- **Frontend** — React.js, Vite
- **PDF Parsing** — pdf.js
- **RAG Engine** — TF-IDF + Cosine Similarity (built from scratch, no libraries)
- **AI Model** — LLaMA 3.1 8B via Groq API
- **Backend** — Node.js, Express.js

## Run Locally

### Prerequisites
- Node.js v18+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Setup
```bash
# Clone the repo
git clone https://github.com/VishnuVardhanMone/rag-zero.git
cd rag-zero

# Install dependencies
npm install
npm install @google/generative-ai groq-sdk express cors dotenv axios pdfjs-dist

# Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start the backend server
node server.js

# In a new terminal, start the frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Project Structure
```
rag-zero/
├── src/
│   ├── App.jsx        # Main UI + RAG engine
│   ├── App.css        # Styling
│   └── main.jsx       # Entry point
├── server.js          # Express backend (Groq API proxy)
├── .env               # API keys (never committed)
└── README.md
```

## Author

**Vishnu Vardhan** — [LinkedIn](https://www.linkedin.com/in/vishnu-vardhan-8636ab31b/) · [GitHub](https://github.com/VishnuVardhanMone)
