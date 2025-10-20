# RAG Chat Interface - AI Agent with Web Tools

A full-stack AI agent powered by **Retrieval Augmented Generation (RAG)** with tool calling capabilities. Query internal documents, search the web (Tavily), or explore academic papers (Arxiv) - all through an elegant chat interface.

## 🎯 What It Does

- **RAG Mode**: Search and answer questions from documents in the `data/` folder
- **Agent Mode**: Intelligently decide which tools to use:
  - 📄 Internal documents (RAG)
  - 🌐 Web search (Tavily)
  - 📚 Academic papers (Arxiv)

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager (install with `pip install uv`)
- OpenAI API key (set in `.env` file)
- Tavily API key (set in `.env` file)

### 1️⃣ Setup Backend

```bash
# Navigate to project root
cd /path/to/project

# Install dependencies using uv
uv sync
```

**Add `.env` file in project root:**

```
OPENAI_API_KEY=xxx
TAVILY_API_KEY=xxx
```

**Add PDF documents:**

```bash
# Place any PDF files in the data/ folder
# They'll be automatically loaded on server startup
cp your-documents.pdf data/
```

### 2️⃣ Start Backend Server

```bash
# From project root
uv run python -m src.main
```

Server runs on: **<http://localhost:8000>**

Check health: **<http://localhost:8000/health>**

### 3️⃣ Start Frontend

```bash
# In a NEW terminal
cd frontend
npm install  # First time only
npm run dev
```

Frontend runs on: **<http://localhost:5173>**

## ⚡ Quick Setup & Run Script
 
**Terminal 1:**

```bash
uv run python -m src.main
```

**Terminal 2:**

```bash
cd frontend && npm run dev
```

## 💬 Usage

1. Open **<http://localhost:5173>** in your browser
2. **Ask a question** in the chat interface
3. **Toggle "Enable Agent with Web Tools"** to switch between modes:
   - ☐ **RAG Only**: Searches only your documents
   - ☑ **Agent Mode**: Uses all available tools

## 📚 Tech Stack

**Backend:**

- FastAPI
- LangChain
- LangGraph
- OpenAI GPT-4
- Tavily & Arxiv APIs

**Frontend:**

- React 19
- TypeScript
- Vite
- React Markdown
- Modern CSS (Evening Theme)
