
import os
import json
import numpy as np
import faiss

from typing import TypedDict, List, Dict, Any
from google import genai
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from docx import Document
from langgraph.graph import StateGraph, START, END


# =====================================================
# 1. CONFIGURATION
# =====================================================

GEMINI_MODEL = "gemini-3.5-flash-lite"

if not os.environ.get("GEMINI_API_KEY"):
    raise EnvironmentError("GEMINI_API_KEY is missing.")

if not os.environ.get("TAVILY_API_KEY"):
    raise EnvironmentError("TAVILY_API_KEY is missing.")


# =====================================================
# 2. GEMINI SERVICE
# =====================================================

class GeminiService:

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def generate(self, prompt, max_output_tokens=1500):

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": max_output_tokens
            }
        )

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        return response.text


# =====================================================
# 3. TAVILY RESEARCH SEARCH
# =====================================================

class ResearchSearch:

    def __init__(self, client):
        self.client = client

    def search(self, query, max_results=5):

        response = self.client.search(
            query=query,
            max_results=max_results,
            search_depth="basic"
        )

        sources = []

        for i, item in enumerate(
            response.get("results", []), 1
        ):
            sources.append({
                "id": f"S{i}",
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })

        return sources


# =====================================================
# 4. SEVEN AI AGENTS
# =====================================================

class OrchestratorAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def plan(self, topic):

        prompt = f"""
        You are the Orchestrator Agent.

        Create a structured research plan for:
        {topic}

        Include:
        1. Research objectives
        2. Important questions
        3. Analysis areas
        4. Expected report sections

        Return a clear structured plan.
        """

        return self.gemini.generate(prompt)


class ResearchAgent:

    def __init__(self, search):
        self.search_engine = search

    def research(self, topic):

        return self.search_engine.search(
            topic,
            max_results=7
        )


class AnalysisAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def analyze(self, topic, sources):

        evidence = json.dumps(sources, ensure_ascii=False)

        prompt = f"""
        You are an AI Research Analysis Agent.

        Topic: {topic}

        Research evidence:
        {evidence}

        Analyze:
        1. Major findings
        2. Important patterns
        3. Advantages
        4. Limitations
        5. Evidence gaps

        Use source IDs when referencing evidence.
        Do not invent facts.
        """

        return self.gemini.generate(prompt, 2000)


class FactCheckerAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def check(self, topic, analysis, sources):

        prompt = f"""
        You are a Fact Checking Agent.

        Topic: {topic}

        Analysis:
        {analysis}

        Available sources:
        {json.dumps(sources, ensure_ascii=False)}

        Review important claims.

        Classify claims as:
        - Supported
        - Partially supported
        - Unsupported
        - Contradicted

        Explain limitations.
        Never claim that an AI check guarantees truth.
        """

        return self.gemini.generate(prompt, 2000)


class SummarizerAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def summarize(self, analysis, fact_check):

        prompt = f"""
        You are a Research Summarization Agent.

        Analysis:
        {analysis}

        Fact checking:
        {fact_check}

        Create a concise executive summary.
        Preserve important caveats and source references.
        """

        return self.gemini.generate(prompt, 1200)


class WriterAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def write_report(
        self, topic, plan, analysis,
        fact_check, summary, sources
    ):

        prompt = f"""
        You are a Professional Research Report Writer.

        Research topic: {topic}

        Research plan:
        {plan}

        Analysis:
        {analysis}

        Fact check:
        {fact_check}

        Executive summary:
        {summary}

        Sources:
        {json.dumps(sources, ensure_ascii=False)}

        Prepare a professional report with:

        1. Title
        2. Executive Summary
        3. Research Objectives
        4. Methodology
        5. Major Findings
        6. Detailed Analysis
        7. Fact Checking and Limitations
        8. Conclusion
        9. References

        Cite available source IDs.
        Never invent references.
        """

        return self.gemini.generate(prompt, 4000)


class QualityAgent:

    def __init__(self, gemini):
        self.gemini = gemini

    def review(self, report):

        prompt = f"""
        You are the Quality Review Agent.

        Review this research report:

        {report}

        Check:
        1. Completeness
        2. Clarity
        3. Logical consistency
        4. Source references
        5. Unsupported claims
        6. Limitations

        Give a quality assessment and improvements.
        Finish with exactly one status:
        APPROVED or NEEDS MAJOR REVISION.
        """

        return self.gemini.generate(prompt, 1500)


# =====================================================
# 5. DOCUMENT LOADER
# =====================================================

class DocumentLoader:

    def load(self, file_path):

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".txt":

            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

        elif extension == ".pdf":

            reader = PdfReader(file_path)

            return "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        elif extension == ".docx":

            document = Document(file_path)

            return "\n".join(
                paragraph.text
                for paragraph in document.paragraphs
            )

        else:
            raise ValueError("Only PDF, TXT and DOCX are supported.")


# =====================================================
# 6. TEXT CHUNKER
# =====================================================

class TextChunker:

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text):

        words = text.split()
        chunks = []

        step = self.chunk_size - self.overlap

        if step <= 0:
            raise ValueError("Overlap must be smaller than chunk size.")

        for start in range(0, len(words), step):

            chunk = words[start:start + self.chunk_size]

            if chunk:
                chunks.append(" ".join(chunk))

        return chunks


# =====================================================
# 7. FAISS VECTOR STORE
# =====================================================

class FAISSVectorStore:

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model
        self.dimension = embedding_model.get_sentence_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add_documents(self, chunks, source_name):

        if not chunks:
            return

        embeddings = self.embedding_model.encode(
            chunks,
            convert_to_numpy=True
        )

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        for i, chunk in enumerate(chunks):
            self.metadata.append({
                "text": chunk,
                "source": source_name,
                "chunk": i + 1
            })

    def search(self, query, top_k=4):

        if self.index.ntotal == 0:
            return []

        embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        embedding = np.asarray(
            embedding,
            dtype="float32"
        )

        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(
            embedding,
            min(top_k, self.index.ntotal)
        )

        results = []

        for score, index in zip(scores[0], indices[0]):

            if index < 0:
                continue

            item = self.metadata[index].copy()
            item["score"] = float(score)

            results.append(item)

        return results


# =====================================================
# 8. RAG ASSISTANT
# =====================================================

class ResearchRAG:

    def __init__(
        self,
        loader,
        chunker,
        vector_store,
        gemini
    ):

        self.loader = loader
        self.chunker = chunker
        self.vector_store = vector_store
        self.gemini = gemini

    def add_document(self, file_path):

        text = self.loader.load(file_path)

        if not text.strip():
            raise ValueError("Document contains no extractable text.")

        chunks = self.chunker.split(text)

        self.vector_store.add_documents(
            chunks,
            os.path.basename(file_path)
        )

        return len(chunks)

    def ask(self, question):

        results = self.vector_store.search(
            question,
            top_k=4
        )

        if not results:
            return "No relevant documents found. Upload a document first."

        context = "\n\n".join(
            f"Source: {item['source']}\n{item['text']}"
            for item in results
        )

        prompt = f"""
        Answer the question using only the supplied context.

        Context:
        {context}

        Question:
        {question}

        If the answer is not available in the context,
        clearly say so.

        Mention the document sources used.
        """

        return self.gemini.generate(prompt, 1500)


# =====================================================
# 9. LANGGRAPH STATE
# =====================================================

class ResearchState(TypedDict, total=False):

    topic: str
    plan: str
    sources: List[Dict[str, Any]]
    analysis: str
    fact_check: str
    summary: str
    final_report: str
    quality_review: str
    revision_count: int
    status: str


# =====================================================
# 10. INITIALIZE API CLIENTS AND AGENTS
# =====================================================

gemini_client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

tavily_client = TavilyClient(
    api_key=os.environ["TAVILY_API_KEY"]
)

gemini_service = GeminiService(
    gemini_client,
    GEMINI_MODEL
)

research_search = ResearchSearch(tavily_client)

orchestrator = OrchestratorAgent(gemini_service)
research_agent = ResearchAgent(research_search)
analysis_agent = AnalysisAgent(gemini_service)
fact_checker = FactCheckerAgent(gemini_service)
summarizer = SummarizerAgent(gemini_service)
writer_agent = WriterAgent(gemini_service)
quality_agent = QualityAgent(gemini_service)


# =====================================================
# 11. LANGGRAPH WORKFLOW NODES
# =====================================================

def orchestrator_node(state):

    topic = state["topic"]

    return {
        "plan": orchestrator.plan(topic),
        "status": "Planning completed"
    }


def research_node(state):

    sources = research_agent.research(state["topic"])

    return {
        "sources": sources,
        "status": "Research completed"
    }


def analysis_node(state):

    analysis = analysis_agent.analyze(
        state["topic"],
        state.get("sources", [])
    )

    return {
        "analysis": analysis,
        "status": "Analysis completed"
    }


def fact_checker_node(state):

    result = fact_checker.check(
        state["topic"],
        state["analysis"],
        state.get("sources", [])
    )

    return {
        "fact_check": result,
        "status": "Fact checking completed"
    }


def summarizer_node(state):

    summary = summarizer.summarize(
        state["analysis"],
        state["fact_check"]
    )

    return {
        "summary": summary,
        "status": "Summary completed"
    }


def writer_node(state):

    report = writer_agent.write_report(
        state["topic"],
        state["plan"],
        state["analysis"],
        state["fact_check"],
        state["summary"],
        state.get("sources", [])
    )

    return {
        "final_report": report,
        "status": "Report generated"
    }


def quality_node(state):

    review = quality_agent.review(
        state["final_report"]
    )

    return {
        "quality_review": review,
        "status": "Quality review completed"
    }


def quality_decision(state):

    review = state.get("quality_review", "").upper()

    if (
        "NEEDS MAJOR REVISION" in review
        and state.get("revision_count", 0) < 1
    ):
        return "revise"

    return "approved"


def revision_node(state):

    revised_report = gemini_service.generate(
        f"""
        Improve this research report using the review.

        Report:
        {state['final_report']}

        Review:
        {state['quality_review']}

        Preserve evidence and references.
        Do not invent facts.
        """,
        4000
    )

    return {
        "final_report": revised_report,
        "revision_count": state.get("revision_count", 0) + 1,
        "status": "Revision completed"
    }


# =====================================================
# 12. BUILD LANGGRAPH
# =====================================================

workflow_builder = StateGraph(ResearchState)

workflow_builder.add_node("orchestrator", orchestrator_node)
workflow_builder.add_node("research", research_node)
workflow_builder.add_node("analysis", analysis_node)
workflow_builder.add_node("fact_checker", fact_checker_node)
workflow_builder.add_node("summarizer", summarizer_node)
workflow_builder.add_node("writer", writer_node)
workflow_builder.add_node("quality", quality_node)
workflow_builder.add_node("revision", revision_node)

workflow_builder.add_edge(START, "orchestrator")
workflow_builder.add_edge("orchestrator", "research")
workflow_builder.add_edge("research", "analysis")
workflow_builder.add_edge("analysis", "fact_checker")
workflow_builder.add_edge("fact_checker", "summarizer")
workflow_builder.add_edge("summarizer", "writer")
workflow_builder.add_edge("writer", "quality")

workflow_builder.add_conditional_edges(
    "quality",
    quality_decision,
    {
        "approved": END,
        "revise": "revision"
    }
)

workflow_builder.add_edge("revision", END)

research_workflow = workflow_builder.compile()


# =====================================================
# 13. INITIALIZE RAG
# =====================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

document_loader = DocumentLoader()

text_chunker = TextChunker(
    chunk_size=500,
    overlap=100
)

vector_store = FAISSVectorStore(
    embedding_model
)

rag_assistant = ResearchRAG(
    document_loader,
    text_chunker,
    vector_store,
    gemini_service
)


print("ResearchMind AI backend initialized successfully.")
