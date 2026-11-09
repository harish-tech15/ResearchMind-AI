# 🧠 ResearchMind AI – Multi-Agent AI Research Assistant

An AI-powered multi-agent research assistant built using Python, Google Gemini, Tavily Search, LangGraph, FAISS, RAG and Streamlit.

ResearchMind AI automates research workflows by collecting information, analyzing findings, checking claims, summarizing results and generating structured research reports.

## 🚀 Features

- **Multi-Agent Research:** Seven specialized AI agents work together.
- **Web Research:** Collects information using Tavily Search API.
- **AI Analysis:** Analyzes research findings using Google Gemini.
- **Fact Checking:** Reviews claims against retrieved research sources.
- **Smart Summarization:** Generates concise research summaries.
- **Report Generation:** Creates structured research reports.
- **RAG Document Chat:** Upload PDF, TXT and DOCX documents and ask questions about their content.
- **Vector Search:** Uses Sentence Transformers and FAISS for document retrieval.
- **Report Export:** Download research reports in PDF and Markdown formats.
- **Interactive Dashboard:** Built with Streamlit.

## 🏗️ Seven-Agent Architecture

| Agent | Responsibility |
|---|---|
| Orchestrator Agent | Plans the research workflow |
| Research Agent | Searches for relevant information |
| Analysis Agent | Analyzes collected information |
| Fact Checker Agent | Checks claims against available sources |
| Summarizer Agent | Creates concise summaries |
| Writer Agent | Generates the final research report |
| Quality Agent | Reviews the report quality |

## 🛠️ Tech Stack

- Python
- Google Gemini API
- Tavily Search API
- LangGraph
- LangChain ecosystem
- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- FAISS
- Streamlit
- PyPDF
- python-docx
- ReportLab

## 📂 Project Structure

```text
ResearchMind-AI/
│
├── app.py
├── researchmind_core.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── uploads/
│   └── .gitkeep
│
└── exports/
    └── .gitkeep
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/ResearchMind-AI.git
```

### 2. Navigate to the project folder

```bash
cd ResearchMind-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
TAVILY_API_KEY = "your_tavily_api_key"
```

Get your API keys:

- Google Gemini: https://aistudio.google.com/
- Tavily: https://tavily.com/

**Security:** Never upload API keys, `.env` files or `secrets.toml` to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 💡 How to Use

1. Launch the ResearchMind AI application.
2. Open the Research tab.
3. Enter your research topic.
4. Generate a research report.
5. Explore the analysis, fact-checking, summary and sources.
6. Download the report as PDF or Markdown.
7. Open Document Chat to upload a PDF, TXT or DOCX file.
8. Ask questions about the uploaded document.

## 🔍 RAG Workflow

```text
Document Upload
      ↓
Document Extraction
      ↓
Text Chunking
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Database
      ↓
Similarity Search
      ↓
Relevant Context
      ↓
Google Gemini
      ↓
AI Generated Answer
```

## 🔄 Multi-Agent Workflow

```text
Research Topic
      ↓
Orchestrator Agent
      ↓
Research Agent
      ↓
Analysis Agent
      ↓
Fact Checker Agent
      ↓
Summarizer Agent
      ↓
Writer Agent
      ↓
Quality Agent
      ↓
Final Research Report
```

## 🔐 Environment Variables

The application requires:

- `GEMINI_API_KEY`
- `TAVILY_API_KEY`

For deployment on Streamlit Community Cloud, configure these values in the application's Secrets settings.

## ⚠️ Limitations

- Research quality depends on the availability and reliability of external sources.
- AI-generated answers may contain errors and should be independently verified.
- The current FAISS document index is in memory and is not permanently persisted.
- Uploaded documents are not retained as a permanent knowledge base.
- The application is a prototype and has not been validated for production-scale deployment.

## 🔮 Future Enhancements

- Persistent vector database.
- Advanced source citations.
- Real-time agent progress tracking.
- User authentication.
- Research history persistence.
- Additional document formats.
- Advanced research evaluation.

## 👨‍💻 Author

**Harish S.**

B.Tech – Artificial Intelligence and Data Science

GitHub: https://github.com/harish-tech15

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

**ResearchMind AI – Turning Information into Intelligence.**
