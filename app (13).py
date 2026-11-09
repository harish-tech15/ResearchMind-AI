
import os
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import streamlit as st

PROJECT_DIR = "/content/ResearchMind_AI"
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")
EXPORT_DIR = os.path.join(PROJECT_DIR, "exports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)

# Load Streamlit Cloud secrets if available
try:
    if "GEMINI_API_KEY" not in os.environ:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

    if "TAVILY_API_KEY" not in os.environ:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide"
)

# -------------------- CUSTOM DESIGN --------------------

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #8bafff;
}
.subtitle {
    color: #9ca3af;
    font-size: 15px;
}
.metric-card {
    background: #172238;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263650;
}
div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------- BACKEND --------------------

@st.cache_resource
def load_backend():
    import researchmind_core
    return researchmind_core

try:
    core = load_backend()
except Exception as error:
    st.error("Backend initialization failed.")
    st.exception(error)
    st.stop()

# -------------------- SESSION STATE --------------------

defaults = {
    "history": [],
    "latest_result": None,
    "latest_topic": "",
    "uploaded_hashes": set(),
    "rag_assistant": None,
    "rag_messages": []
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Create a separate vector store for this user's session
if st.session_state.rag_assistant is None:
    session_vector_store = core.FAISSVectorStore(
        core.embedding_model
    )

    st.session_state.rag_assistant = core.ResearchRAG(
        core.document_loader,
        core.text_chunker,
        session_vector_store,
        core.gemini_service
    )

rag = st.session_state.rag_assistant

# -------------------- HELPER FUNCTIONS --------------------

def make_markdown(topic, result):
    return f"""
# ResearchMind AI Research Report

**Topic:** {topic}

**Generated:** {datetime.now().strftime("%d-%m-%Y %H:%M")}

---

## Final Report

{result.get("final_report", "")}

---

## Analysis

{result.get("analysis", "")}

---

## Fact Check

{result.get("fact_check", "")}

---

## Summary

{result.get("summary", "")}

---

## Quality Review

{result.get("quality_review", "")}

---

## Sources

{format_sources(result.get("sources", []))}
"""


def format_sources(sources):
    if not sources:
        return "No sources available."

    lines = []
    for index, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        lines.append(f"{index}. [{title}]({url})")

    return "\n".join(lines)


def make_pdf(markdown_text):
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer
    )
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from io import BytesIO

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()
    styles["Title"].textColor = colors.HexColor("#244b87")
    styles["Heading2"].textColor = colors.HexColor("#244b87")

    story = []

    for line in markdown_text.splitlines():
        line = line.strip()

        if not line or line == "---":
            story.append(Spacer(1, 7))
            continue

        if line.startswith("# "):
            story.append(Paragraph(
                escape(line[2:]), styles["Title"]
            ))
        elif line.startswith("## "):
            story.append(Paragraph(
                escape(line[3:]), styles["Heading2"]
            ))
        else:
            clean = line.replace("**", "")
            clean = clean.replace("`", "")
            clean = clean.replace("[", "").replace("]", "")
            clean = clean.replace("(", " ").replace(")", "")
            story.append(Paragraph(
                escape(clean), styles["BodyText"]
            ))
            story.append(Spacer(1, 5))

    doc.build(story)
    return buffer.getvalue()


def show_sources(sources):
    if not sources:
        st.info("No research sources returned.")
        return

    for i, source in enumerate(sources, 1):
        with st.expander(
            f"{i}. {source.get('title', 'Untitled source')}"
        ):
            st.write(source.get("content", ""))
            if source.get("url"):
                st.markdown(
                    f"[Open source]({source['url']})"
                )

# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.markdown("## 🧠 ResearchMind AI")
    st.caption("Multi-Agent AI Research Assistant")

    st.divider()

    st.markdown("### System Status")

    st.success("Backend initialized")

    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini API configured")
    else:
        st.error("Gemini API key missing")

    if os.environ.get("TAVILY_API_KEY"):
        st.success("Tavily API configured")
    else:
        st.error("Tavily API key missing")

    st.info("RAG assistant initialized")

    st.divider()

    st.markdown("### Seven AI Agents")
    agent_names = [
        "1. Orchestrator",
        "2. Research",
        "3. Analysis",
        "4. Fact Checker",
        "5. Summarizer",
        "6. Writer",
        "7. Quality"
    ]

    for agent in agent_names:
        st.write("🤖 " + agent)

    st.divider()
    st.caption("ResearchMind AI | Prototype")

# -------------------- HEADER --------------------

st.markdown(
    '<div class="main-title">🧠 ResearchMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your intelligent multi-agent research workspace</div>',
    unsafe_allow_html=True
)

st.write("")

# -------------------- DASHBOARD --------------------

tab_research, tab_documents, tab_history, tab_about = st.tabs([
    "🔍 Research",
    "📚 Document Chat",
    "🕘 History",
    "ℹ️ About"
])

# =====================================================
# TAB 1 - RESEARCH
# =====================================================

with tab_research:

    st.subheader("AI Research Workspace")

    topic = st.text_area(
        "Enter your research topic",
        placeholder=(
            "Example: Explain RAG architecture, "
            "applications, advantages and limitations."
        ),
        height=120
    )

    run_research = st.button(
        "🚀 Generate Research Report",
        type="primary",
        use_container_width=True
    )

    if run_research:

        if not topic.strip():
            st.warning("Please enter a research topic.")

        elif not os.environ.get("GEMINI_API_KEY"):
            st.error("Gemini API key is not configured.")

        elif not os.environ.get("TAVILY_API_KEY"):
            st.error("Tavily API key is not configured.")

        else:
            try:
                with st.spinner(
                    "ResearchMind AI agents are working..."
                ):
                    result = core.research_workflow.invoke({
                        "topic": topic.strip(),
                        "revision_count": 0
                    })

                st.session_state.latest_result = result
                st.session_state.latest_topic = topic.strip()

                st.session_state.history.insert(0, {
                    "topic": topic.strip(),
                    "result": result,
                    "time": datetime.now().strftime(
                        "%d-%m-%Y %H:%M"
                    )
                })

                st.success("Research workflow completed!")

            except Exception as error:
                st.error("Research workflow failed.")
                st.exception(error)

    # Display latest research
    result = st.session_state.latest_result

    if result:

        st.divider()
        st.subheader("Research Results")

        stages = [
            "Orchestrator",
            "Research",
            "Analysis",
            "Fact Checker",
            "Summarizer",
            "Writer",
            "Quality"
        ]

        st.markdown("### Agent Workflow")

        cols = st.columns(4)

        for index, name in enumerate(stages):
            with cols[index % 4]:
                st.success("✓ " + name)

        st.caption(
            "Stages shown after the workflow returns successfully; "
            "these are not live streamed progress updates."
        )

        st.divider()

        result_tabs = st.tabs([
            "📄 Final Report",
            "📊 Analysis",
            "🔎 Fact Check",
            "📝 Summary",
            "🌐 Sources",
            "⭐ Quality"
        ])

        with result_tabs[0]:
            st.markdown(
                result.get("final_report", "No report generated.")
            )

        with result_tabs[1]:
            st.markdown(
                result.get("analysis", "No analysis available.")
            )

        with result_tabs[2]:
            st.markdown(
                result.get("fact_check", "No fact check available.")
            )

        with result_tabs[3]:
            st.markdown(
                result.get("summary", "No summary available.")
            )

        with result_tabs[4]:
            show_sources(result.get("sources", []))

        with result_tabs[5]:
            st.markdown(
                result.get("quality_review", "No quality review.")
            )

        st.divider()
        st.subheader("Export Research")

        markdown_report = make_markdown(
            st.session_state.latest_topic,
            result
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇️ Download Markdown",
                data=markdown_report,
                file_name="ResearchMind_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

        with col2:
            try:
                pdf_data = make_pdf(markdown_report)

                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_data,
                    file_name="ResearchMind_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as error:
                st.warning(f"PDF export unavailable: {error}")

# =====================================================
# TAB 2 - DOCUMENT CHAT / RAG
# =====================================================

with tab_documents:

    st.subheader("📚 Chat with Your Documents")

    st.write(
        "Upload PDF, TXT or DOCX files and ask questions "
        "based on their content."
    )

    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            if file_hash not in st.session_state.uploaded_hashes:

                extension = Path(uploaded_file.name).suffix.lower()
                safe_name = file_hash + extension
                file_path = os.path.join(UPLOAD_DIR, safe_name)

                try:
                    with open(file_path, "wb") as file:
                        file.write(file_bytes)

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        chunk_count = rag.add_document(file_path)

                    st.session_state.uploaded_hashes.add(file_hash)

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{chunk_count} chunks indexed."
                    )

                except Exception as error:
                    st.error(
                        f"Could not process {uploaded_file.name}"
                    )
                    st.exception(error)

            else:
                st.info(
                    f"{uploaded_file.name} is already indexed "
                    "in this session."
                )

    st.divider()
    st.markdown("### Ask a Question")

    with st.form("document_question_form"):
        question = st.text_input(
            "Your question",
            placeholder="What does this document explain?"
        )

        ask_button = st.form_submit_button(
            "💬 Ask ResearchMind AI"
        )

    if ask_button:

        if not question.strip():
            st.warning("Enter a question first.")

        elif not st.session_state.uploaded_hashes:
            st.warning("Upload and process a document first.")

        else:
            try:
                with st.spinner("Searching your documents..."):
                    answer = rag.ask(question.strip())

                st.session_state.rag_messages.append({
                    "question": question.strip(),
                    "answer": answer
                })

            except Exception as error:
                st.error("Document question failed.")
                st.exception(error)

    for message in reversed(st.session_state.rag_messages):
        st.markdown("**You:** " + message["question"])
        st.markdown("**ResearchMind AI:**")
        st.write(message["answer"])
        st.divider()

    st.caption(
        "Uploaded documents are held in the current app process/session. "
        "They are not permanently saved in a persistent vector database."
    )

# =====================================================
# TAB 3 - HISTORY
# =====================================================

with tab_history:

    st.subheader("🕘 Research History")

    if not st.session_state.history:
        st.info("Your research history will appear here.")

    else:
        for index, item in enumerate(
            st.session_state.history
        ):
            with st.expander(
                f"{item['topic']} | {item['time']}"
            ):
                st.markdown(
                    item["result"].get(
                        "final_report",
                        "No report available."
                    )
                )

                history_md = make_markdown(
                    item["topic"],
                    item["result"]
                )

                st.download_button(
                    "Download this report",
                    data=history_md,
                    file_name=f"Research_Report_{index + 1}.md",
                    mime="text/markdown",
                    key=f"history_download_{index}"
                )

# =====================================================
# TAB 4 - ABOUT
# =====================================================

with tab_about:

    st.subheader("About ResearchMind AI")

    st.write("""
    ResearchMind AI is a multi-agent research assistant designed
    to automate research workflows.

    **Core technologies**

    - Python
    - Google Gemini API
    - Tavily Search API
    - LangGraph
    - Sentence Transformers
    - FAISS Vector Database
    - Streamlit

    **Seven-agent architecture**

    1. Orchestrator Agent
    2. Research Agent
    3. Analysis Agent
    4. Fact Checker Agent
    5. Summarizer Agent
    6. Writer Agent
    7. Quality Agent

    **Main capabilities**

    - AI-powered web research
    - Multi-agent report generation
    - Fact checking
    - Document-based RAG question answering
    - PDF and Markdown report exports
    """)

    st.info(
        "ResearchMind AI is a prototype. "
        "Always independently verify research findings and sources."
    )

st.divider()

st.caption(
    "ResearchMind AI | Built with Python, Gemini, LangGraph, "
    "FAISS and Streamlit"
)
