
import os
import sys
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from xml.sax.saxutils import escape
from io import BytesIO

import streamlit as st

# =====================================================
# 1. APPLICATION CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ResearchMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GitHub repository directory (read-only on Streamlit Cloud)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use writable temporary storage instead of repository directory
DATA_DIR = os.path.join(
    tempfile.gettempdir(),
    "researchmind_ai"
)

UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)

# =====================================================
# 2. API CONFIGURATION
# =====================================================

for key in ["GEMINI_API_KEY", "TAVILY_API_KEY"]:
    if not os.environ.get(key):
        try:
            value = st.secrets[key]
            if value:
                os.environ[key] = str(value)
        except Exception:
            pass

# =====================================================
# 3. CUSTOM UI
# =====================================================

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

div.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    min-height: 42px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. LOAD BACKEND
# =====================================================

@st.cache_resource
def load_backend():
    import researchmind_core
    return researchmind_core

try:
    core = load_backend()
except Exception as error:
    st.error("ResearchMind AI backend initialization failed.")
    st.exception(error)
    st.stop()

# =====================================================
# 5. SESSION STATE
# =====================================================

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

# Create a separate RAG store for each user session
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

# =====================================================
# 6. HELPER FUNCTIONS
# =====================================================

def format_sources(sources):

    if not sources:
        return "No sources available."

    lines = []

    for index, source in enumerate(sources, 1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")

        if url:
            lines.append(f"{index}. [{title}]({url})")
        else:
            lines.append(f"{index}. {title}")

    return "\n".join(lines)


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


def make_pdf(markdown_text):

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer
    )

    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors

    buffer = BytesIO()

    document = SimpleDocTemplate(
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
            story.append(
                Paragraph(
                    escape(line[2:]),
                    styles["Title"]
                )
            )

        elif line.startswith("## "):
            story.append(
                Paragraph(
                    escape(line[3:]),
                    styles["Heading2"]
                )
            )

        else:
            clean = line.replace("**", "")
            clean = clean.replace("`", "")

            story.append(
                Paragraph(
                    escape(clean),
                    styles["BodyText"]
                )
            )

            story.append(Spacer(1, 5))

    document.build(story)

    return buffer.getvalue()


def show_sources(sources):

    if not sources:
        st.info("No research sources returned.")
        return

    for index, source in enumerate(sources, 1):

        title = source.get("title", "Untitled source")

        with st.expander(f"{index}. {title}"):

            st.write(source.get("content", ""))

            if source.get("url"):
                st.markdown(
                    f"[Open Source]({source['url']})"
                )


# =====================================================
# 7. SIDEBAR
# =====================================================

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

    st.success("RAG assistant initialized")

    st.divider()

    st.markdown("### Seven AI Agents")

    agents = [
        "Orchestrator",
        "Research",
        "Analysis",
        "Fact Checker",
        "Summarizer",
        "Writer",
        "Quality"
    ]

    for agent in agents:
        st.write("🤖 " + agent)

    st.divider()

    st.caption("ResearchMind AI | Prototype")


# =====================================================
# 8. MAIN HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🧠 ResearchMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Intelligent Multi-Agent Research Workspace'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# =====================================================
# 9. APPLICATION TABS
# =====================================================

tab_research, tab_documents, tab_history, tab_about = st.tabs([
    "🔍 Research",
    "📚 Document Chat",
    "🕘 History",
    "ℹ️ About"
])

# =====================================================
# 10. RESEARCH WORKSPACE
# =====================================================

with tab_research:

    st.subheader("AI Research Workspace")

    topic = st.text_area(
        "Enter your research topic",
        placeholder="Example: Explain RAG architecture and applications.",
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

                st.session_state.history.insert(
                    0,
                    {
                        "topic": topic.strip(),
                        "result": result,
                        "time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M"
                        )
                    }
                )

                st.success("Research workflow completed!")

            except Exception as error:

                st.error("Research workflow failed.")
                st.exception(error)

    # DISPLAY RESULTS

    result = st.session_state.latest_result

    if result:

        st.divider()

        st.subheader("Research Results")

        st.markdown("### Agent Workflow")

        stages = [
            "Orchestrator",
            "Research",
            "Analysis",
            "Fact Checker",
            "Summarizer",
            "Writer",
            "Quality"
        ]

        columns = st.columns(4)

        for index, name in enumerate(stages):

            with columns[index % 4]:
                st.success("✓ " + name)

        st.caption(
            "Workflow stages shown after successful completion. "
            "This is not live streamed progress."
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
                result.get(
                    "final_report",
                    "No report generated."
                )
            )

        with result_tabs[1]:

            st.markdown(
                result.get(
                    "analysis",
                    "No analysis available."
                )
            )

        with result_tabs[2]:

            st.markdown(
                result.get(
                    "fact_check",
                    "No fact check available."
                )
            )

        with result_tabs[3]:

            st.markdown(
                result.get(
                    "summary",
                    "No summary available."
                )
            )

        with result_tabs[4]:

            show_sources(result.get("sources", []))

        with result_tabs[5]:

            st.markdown(
                result.get(
                    "quality_review",
                    "No quality review available."
                )
            )

        # EXPORT

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
# 11. DOCUMENT CHAT - RAG
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

            file_hash = hashlib.sha256(
                file_bytes
            ).hexdigest()

            if file_hash not in st.session_state.uploaded_hashes:

                extension = Path(
                    uploaded_file.name
                ).suffix.lower()

                safe_name = file_hash + extension

                file_path = os.path.join(
                    UPLOAD_DIR,
                    safe_name
                )

                try:

                    with open(file_path, "wb") as file:
                        file.write(file_bytes)

                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):

                        chunk_count = rag.add_document(file_path)

                    st.session_state.uploaded_hashes.add(
                        file_hash
                    )

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

            st.warning("Upload a document first.")

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

    for message in reversed(
        st.session_state.rag_messages
    ):

        st.markdown(
            "**You:** " + message["question"]
        )

        st.markdown("**ResearchMind AI:**")

        st.write(message["answer"])

        st.divider()

    st.caption(
        "Documents are stored temporarily. They may be lost "
        "when the application restarts."
    )


# =====================================================
# 12. RESEARCH HISTORY
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

                history_markdown = make_markdown(
                    item["topic"],
                    item["result"]
                )

                st.download_button(
                    "Download this report",
                    data=history_markdown,
                    file_name=f"Research_Report_{index + 1}.md",
                    mime="text/markdown",
                    key=f"history_download_{index}"
                )


# =====================================================
# 13. ABOUT
# =====================================================

with tab_about:

    st.subheader("About ResearchMind AI")

    st.write("""
    ResearchMind AI is a multi-agent AI research assistant
    designed to automate research workflows.

    **Core Technologies**

    - Python
    - Google Gemini API
    - Tavily Search API
    - LangGraph
    - Sentence Transformers
    - FAISS Vector Database
    - Retrieval-Augmented Generation
    - Streamlit

    **Seven-Agent Architecture**

    1. Orchestrator Agent
    2. Research Agent
    3. Analysis Agent
    4. Fact Checker Agent
    5. Summarizer Agent
    6. Writer Agent
    7. Quality Agent

    **Main Capabilities**

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


# =====================================================
# 14. FOOTER
# =====================================================

st.divider()

st.caption(
    "ResearchMind AI | Built with Python, Gemini, "
    "LangGraph, FAISS and Streamlit"
)
