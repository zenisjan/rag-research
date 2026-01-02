"""
RAG Actor for querying Pinecone vector store with meeting notes context.
Supports Standby mode with HTTP request/response handling.
"""
import asyncio
import json
import os
import re
import math
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any, Tuple
from urllib.parse import parse_qs, urlparse

from apify import Actor
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


# Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
DEFAULT_K = 10
DEFAULT_THRESHOLD = 0.6
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RECENCY_DECAY_DAYS = 180
HTTP_PORT = int(os.getenv('ACTOR_STANDBY_PORT', '8080'))


@dataclass
class Config:
    """Validated input configuration."""
    openai_api_key: str
    pinecone_api_key: str
    index_name: str
    question: str
    k: int
    threshold: float
    start_template: str
    recency_weight: float
    recency_decay_days: int


def get_config(input_data: dict) -> Config:
    """
    Validate inputs. Environment variables take priority over input data.
    """
    def get_value(*keys, env_var: str = None, default=None):
        """Get value from env or input data with fallback."""
        if env_var:
            val = os.getenv(env_var)
            if val and val.strip():
                return val.strip()
        for key in keys:
            val = input_data.get(key)
            if val and str(val).strip():
                return str(val).strip()
        return default

    openai_api_key = get_value('openai_key', 'openai_api_key', env_var='OPENAI_API_KEY')
    pinecone_api_key = get_value('pinecone_key', 'pinecone_api_key', env_var='PINECONE_API_KEY')
    index_name = get_value('index_name', 'index', env_var='INDEX_NAME')
    question = get_value('question', 'query', env_var='QUESTION')

    # Parse numeric values
    k_raw = get_value('k', env_var='K', default=str(DEFAULT_K))
    try:
        k = max(1, min(50, int(k_raw)))
    except (ValueError, TypeError):
        k = DEFAULT_K

    threshold_raw = get_value('threshold', env_var='THRESHOLD', default=str(DEFAULT_THRESHOLD))
    try:
        threshold = max(0.0, min(1.0, float(threshold_raw)))
    except (ValueError, TypeError):
        threshold = DEFAULT_THRESHOLD

    recency_weight_raw = get_value('recency_weight', env_var='RECENCY_WEIGHT', default=str(DEFAULT_RECENCY_WEIGHT))
    try:
        recency_weight = max(0.0, min(1.0, float(recency_weight_raw)))
    except (ValueError, TypeError):
        recency_weight = DEFAULT_RECENCY_WEIGHT

    recency_decay_raw = get_value('recency_decay_days', env_var='RECENCY_DECAY_DAYS', default=str(DEFAULT_RECENCY_DECAY_DAYS))
    try:
        recency_decay_days = max(1, int(recency_decay_raw))
    except (ValueError, TypeError):
        recency_decay_days = DEFAULT_RECENCY_DECAY_DAYS

    start_template = get_value(
        'start_template',
        env_var='START_TEMPLATE',
        default='Answer the question based on the context provided.'
    )

    # Validate required fields
    errors = []
    if not openai_api_key:
        errors.append("OpenAI API key is required (OPENAI_API_KEY or 'openai_key')")
    if not pinecone_api_key:
        errors.append("Pinecone API key is required (PINECONE_API_KEY or 'pinecone_key')")
    if not index_name:
        errors.append("Index name is required (INDEX_NAME or 'index_name')")
    if not question:
        errors.append("Question is required (QUESTION or 'question')")

    if errors:
        raise ValueError("; ".join(errors))

    return Config(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        question=question,
        k=k,
        threshold=threshold,
        start_template=start_template,
        recency_weight=recency_weight,
        recency_decay_days=recency_decay_days,
    )


def format_context(documents: List[Document]) -> str:
    """Format documents with source metadata for the LLM context."""
    parts = []
    for idx, doc in enumerate(documents, 1):
        url = doc.metadata.get('page_url', 'N/A')
        date = doc.metadata.get('page_date', 'N/A')

        source_info = f"[Source {idx}]"
        if url != 'N/A':
            source_info += f" URL: {url}"
        if date != 'N/A':
            source_info += f" | Date: {date}"

        parts.append(f"{source_info}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def extract_citations(response: str, documents: List[Document]) -> List[Dict[str, Any]]:
    """Extract source citations from the response."""
    citations = []
    seen = set()

    for match in re.finditer(r'\[Source\s*(\d+)\]|\[(\d+)\]', response):
        num = int(match.group(1) or match.group(2))
        if 1 <= num <= len(documents) and num not in seen:
            seen.add(num)
            doc = documents[num - 1]
            citations.append({
                'source_number': num,
                'notion_url': doc.metadata.get('page_url'),
                'date': doc.metadata.get('page_date'),
            })

    return citations


def parse_date(date_str: str) -> datetime | None:
    """Parse date string from various formats."""
    if not date_str or date_str == 'N/A':
        return None

    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def calculate_recency_score(page_date: str, decay_days: int) -> float:
    """
    Calculate recency score using exponential decay.
    Returns a score between 0 and 1.
    """
    parsed_date = parse_date(page_date)
    if not parsed_date:
        return 0.5

    now = datetime.now()
    if parsed_date.tzinfo is not None:
        parsed_date = parsed_date.replace(tzinfo=None)

    age_days = (now - parsed_date).days

    if age_days < 0:
        return 1.0

    return math.pow(2, -age_days / decay_days)


def calculate_combined_score(
    similarity_score: float,
    recency_score: float,
    recency_weight: float
) -> float:
    """Combine similarity and recency scores."""
    return (1 - recency_weight) * similarity_score + recency_weight * recency_score


def retrieve_documents_sync(
    vector_store: PineconeVectorStore,
    question: str,
    k: int,
    threshold: float,
    recency_weight: float = 0.0,
    recency_decay_days: int = 180
) -> List[Document]:
    """
    Retrieve relevant documents with recency-aware ranking (synchronous version).
    """
    fetch_multiplier = 2 if recency_weight > 0 else 1
    fetch_k = min(k * fetch_multiplier, 50)

    results_with_scores: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_score(question, k=fetch_k)
    )

    if not results_with_scores:
        return []

    scored_docs: List[Tuple[Document, float, float, float]] = []

    for doc, similarity_score in results_with_scores:
        if similarity_score < threshold:
            continue

        page_date = doc.metadata.get('page_date', '')
        recency_score = calculate_recency_score(page_date, recency_decay_days)
        combined_score = calculate_combined_score(
            similarity_score, recency_score, recency_weight
        )

        scored_docs.append((doc, combined_score, similarity_score, recency_score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    top_docs = scored_docs[:k]

    if top_docs and recency_weight > 0:
        Actor.log.debug(f"Recency weight: {recency_weight}, decay days: {recency_decay_days}")
        for doc, combined, sim, rec in top_docs[:3]:
            date = doc.metadata.get('page_date', 'N/A')
            Actor.log.debug(
                f"Doc date={date}: similarity={sim:.3f}, recency={rec:.3f}, combined={combined:.3f}"
            )

    return [doc for doc, _, _, _ in top_docs]


async def retrieve_documents(
    vector_store: PineconeVectorStore,
    question: str,
    k: int,
    threshold: float,
    recency_weight: float = 0.0,
    recency_decay_days: int = 180
) -> List[Document]:
    """Async wrapper for retrieve_documents_sync."""
    return await asyncio.to_thread(
        retrieve_documents_sync,
        vector_store, question, k, threshold, recency_weight, recency_decay_days
    )


def process_rag_query(input_data: dict) -> dict:
    """
    Process a RAG query and return the result.
    This is the core logic extracted for reuse in both batch and HTTP modes.
    """
    try:
        config = get_config(input_data)

        Actor.log.info(f"Question: {config.question[:100]}...")
        Actor.log.info(f"Settings: k={config.k}, threshold={config.threshold}, recency_weight={config.recency_weight}")

        # Initialize services
        embeddings = OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=EMBEDDING_MODEL,
        )

        pinecone_client = Pinecone(api_key=config.pinecone_api_key)
        index = pinecone_client.Index(config.index_name)

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
        )

        llm = ChatOpenAI(
            openai_api_key=config.openai_api_key,
            model=LLM_MODEL,
            temperature=0.0,
        )

        # Retrieve context with recency-aware ranking
        documents = retrieve_documents_sync(
            vector_store,
            config.question,
            config.k,
            config.threshold,
            config.recency_weight,
            config.recency_decay_days,
        )
        Actor.log.info(f"Retrieved {len(documents)} documents")

        if not documents:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "citations": [],
                "sources_used": 0,
            }

        # Build and execute chain
        context = format_context(documents)

        prompt = ChatPromptTemplate.from_template("""{start_template}

Context from meeting recordings:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer in markdown format
- Cite sources using [Source X] format
- Include Notion URLs and dates when citing
- Only say "I don't know" if context has no relevant information
""")

        chain = (
            {
                "start_template": lambda _: config.start_template,
                "context": lambda _: context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(config.question)
        citations = extract_citations(answer, documents)

        result = {
            "answer": answer,
            "citations": citations,
            "sources_used": len(documents),
        }

        Actor.log.info(f"Answer: {answer}")
        Actor.log.info(f"Citations: {citations}")
        Actor.log.info(f"Generated answer with {len(citations)} citations")
        return result

    except ValueError as e:
        Actor.log.error(f"Validation error: {e}")
        return {"error": str(e), "error_type": "validation"}
    except Exception as e:
        Actor.log.exception(f"Unexpected error: {e}")
        return {"error": str(e), "error_type": "unexpected"}


class RAGRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RAG queries in Standby mode."""

    def log_message(self, format: str, *args) -> None:
        """Override to use Actor.log instead of stderr."""
        Actor.log.debug(f"HTTP: {format % args}")

    def send_json_response(self, data: dict, status_code: int = 200) -> None:
        """Send a JSON response."""
        response_body = json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self) -> None:
        """Handle GET requests."""
        # Handle Apify standby readiness probe
        if 'x-apify-container-server-readiness-probe' in self.headers:
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Readiness probe OK')
            return

        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/' or path == '/health':
            self.send_json_response({
                "status": "ready",
                "message": "Interviews RAG Actor is running in Standby mode",
                "endpoints": {
                    "POST /query": "Submit a RAG query",
                    "GET /health": "Health check",
                }
            })
            return

        if path == '/query':
            # Handle query via GET with query parameters
            query_params = parse_qs(parsed_url.query)
            input_data = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}

            if not input_data.get('question'):
                self.send_json_response(
                    {"error": "Missing 'question' parameter", "error_type": "validation"},
                    status_code=400
                )
                return

            result = process_rag_query(input_data)
            status_code = 200 if 'error' not in result else 400
            self.send_json_response(result, status_code)
            return

        self.send_json_response(
            {"error": f"Not found: {path}", "error_type": "not_found"},
            status_code=404
        )

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/query' or path == '/':
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_json_response(
                    {"error": "Empty request body", "error_type": "validation"},
                    status_code=400
                )
                return

            try:
                body = self.rfile.read(content_length)
                input_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                self.send_json_response(
                    {"error": f"Invalid JSON: {e}", "error_type": "validation"},
                    status_code=400
                )
                return

            result = process_rag_query(input_data)
            status_code = 200 if 'error' not in result else 400
            self.send_json_response(result, status_code)
            return

        self.send_json_response(
            {"error": f"Not found: {path}", "error_type": "not_found"},
            status_code=404
        )


def run_http_server() -> None:
    """Run the HTTP server for Standby mode."""
    server_address = ('', HTTP_PORT)
    httpd = HTTPServer(server_address, RAGRequestHandler)
    Actor.log.info(f"Starting HTTP server on port {HTTP_PORT}")
    Actor.log.info("Standby mode enabled - waiting for requests...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        Actor.log.info("Shutting down HTTP server...")
        httpd.shutdown()


async def run_batch_mode() -> None:
    """Run in traditional batch mode (process single input and exit)."""
    input_data = await Actor.get_input() or {}
    result = process_rag_query(input_data)
    await Actor.push_data(result)


async def main() -> None:
    """Main entry point for the Apify Actor."""
    async with Actor:
        # Check if running in Standby mode
        is_standby = os.getenv('ACTOR_STANDBY_PORT') is not None

        if is_standby:
            Actor.log.info("Running in Standby mode")
            # Run HTTP server in a thread to not block the async loop
            await asyncio.to_thread(run_http_server)
        else:
            Actor.log.info("Running in batch mode")
            await run_batch_mode()
