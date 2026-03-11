"""
RAG Actor for querying Pinecone vector store with meeting notes context.
Runs only in Standby mode with HTTP request/response handling.
Configuration is loaded from environment variables.
"""
import asyncio
import json
import os
import re
import math
from dataclasses import dataclass
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import List, Dict, Any, Tuple, Optional
from urllib.parse import parse_qs, urlparse

from apify import Actor
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load .env file for local development
load_dotenv()

# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
HTTP_PORT = int(os.getenv('ACTOR_STANDBY_PORT', '8080'))


def is_running_on_apify() -> bool:
    """Check if we're running on Apify platform."""
    return os.getenv('APIFY_IS_AT_HOME') == '1'


def censor_key(key: Optional[str]) -> str:
    """Censor an API key for safe logging."""
    if not key:
        return "(not set)"
    if len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class Config:
    """Configuration loaded from environment variables."""
    openai_api_key: str
    pinecone_api_key: str
    index_name: str
    namespace: str
    k: int
    threshold: float
    start_template: str
    recency_weight: float
    recency_decay_days: int


def get_env_int(name: str, default: int, min_val: int = 1, max_val: int = 1000) -> int:
    """Get integer from environment variable with bounds."""
    try:
        val = os.getenv(name)
        if val:
            return max(min_val, min(max_val, int(val)))
    except (ValueError, TypeError):
        pass
    return default


def get_env_float(name: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Get float from environment variable with bounds."""
    try:
        val = os.getenv(name)
        if val:
            return max(min_val, min(max_val, float(val)))
    except (ValueError, TypeError):
        pass
    return default


def load_config() -> Config:
    """Load configuration from environment variables."""
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    index_name = os.getenv('INDEX_NAME')

    # Validate required fields
    missing = []
    if not openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not index_name:
        missing.append("INDEX_NAME")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Config(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        namespace=os.getenv('NAMESPACE', ''),  # Empty string = default namespace
        k=get_env_int('K', default=20, min_val=1, max_val=50),
        threshold=get_env_float('THRESHOLD', default=0.2),
        start_template=os.getenv('START_TEMPLATE', 'System Instruction (recommended): You are an AI assistant that answers questions strictly based on the retrieved context from Apify customer meeting transcipts and its summaries. Task Instruction: Answer the question only using the provided context from retrieved documents. Do not use external knowledge unless the question explicitly asks about general concepts unrelated to Apify. If the context contains relevant information: Provide a clear and concise answer Include citations to Notion pages links containing the transcripts to all relevant meeting sources used to form your answer If multiple meetings mention the answer, list all relevant links If the context does NOT contain enough information to answer: Respond exactly with: "I do not" know Company Context for Model Awareness (not for output): Apify is a platform for web scraping, automation, and AI agents. Applications running on Apify are called Actors.'),
        recency_weight=get_env_float('RECENCY_WEIGHT', default=0.3),
        recency_decay_days=get_env_int('RECENCY_DECAY_DAYS', default=180, min_val=1, max_val=3650),
    )


# Global config loaded at startup
_config: Optional[Config] = None


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
    """Calculate recency score using exponential decay. Returns 0-1."""
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


def retrieve_documents(
    vector_store: PineconeVectorStore,
    question: str,
    k: int,
    threshold: float,
    recency_weight: float = 0.0,
    recency_decay_days: int = 180
) -> List[Document]:
    """Retrieve relevant documents with recency-aware ranking."""
    fetch_multiplier = 2 if recency_weight > 0 else 1
    fetch_k = min(k * fetch_multiplier, 50)

    Actor.log.info(f"Searching Pinecone with k={fetch_k}, threshold={threshold}")

    results_with_scores: List[Tuple[Document, float]] = (
        vector_store.similarity_search_with_score(question, k=fetch_k)
    )

    Actor.log.info(f"Pinecone returned {len(results_with_scores)} results")

    if not results_with_scores:
        Actor.log.warning("No results returned from Pinecone")
        return []

    # Log raw scores to debug threshold filtering
    if results_with_scores:
        scores = [score for _, score in results_with_scores]
        Actor.log.info(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}")
        Actor.log.info(f"First 5 scores: {[f'{s:.4f}' for s in scores[:5]]}")

    scored_docs: List[Tuple[Document, float, float, float]] = []
    filtered_count = 0

    for doc, similarity_score in results_with_scores:
        if similarity_score < threshold:
            filtered_count += 1
            continue

        page_date = doc.metadata.get('page_date', '')
        recency_score = calculate_recency_score(page_date, recency_decay_days)
        combined_score = calculate_combined_score(
            similarity_score, recency_score, recency_weight
        )

        scored_docs.append((doc, combined_score, similarity_score, recency_score))

    if filtered_count > 0:
        Actor.log.info(f"Filtered out {filtered_count} docs below threshold {threshold}")

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


def process_rag_query(question: str, request_overrides: Optional[Dict[str, Any]] = None) -> dict:
    """
    Process a RAG query and return the result.
    
    Args:
        question: The user's question
        request_overrides: Optional per-request overrides (k, threshold, api keys, etc.)
    """
    global _config

    if not _config:
        return {"error": "Configuration not loaded", "error_type": "config"}

    try:
        Actor.log.info("-" * 40)
        Actor.log.info("Processing RAG query...")

        # Apply request-level overrides if provided
        k = _config.k
        threshold = _config.threshold
        recency_weight = _config.recency_weight
        recency_decay_days = _config.recency_decay_days
        start_template = _config.start_template
        
        # API keys and model - can be overridden per-request
        openai_api_key = _config.openai_api_key
        pinecone_api_key = _config.pinecone_api_key
        index_name = _config.index_name
        llm_model = LLM_MODEL

        if request_overrides:
            if 'k' in request_overrides:
                try:
                    k = max(1, min(50, int(request_overrides['k'])))
                except (ValueError, TypeError):
                    pass
            if 'threshold' in request_overrides:
                try:
                    threshold = max(0.0, min(1.0, float(request_overrides['threshold'])))
                except (ValueError, TypeError):
                    pass
            if 'recency_weight' in request_overrides:
                try:
                    recency_weight = max(0.0, min(1.0, float(request_overrides['recency_weight'])))
                except (ValueError, TypeError):
                    pass
            if 'recency_decay_days' in request_overrides:
                try:
                    recency_decay_days = max(1, int(request_overrides['recency_decay_days']))
                except (ValueError, TypeError):
                    pass
            if 'start_template' in request_overrides:
                start_template = str(request_overrides['start_template'])
            
            # API keys and configuration overrides
            if 'openai_api_key' in request_overrides and request_overrides['openai_api_key']:
                openai_api_key = str(request_overrides['openai_api_key'])
                Actor.log.info(f"Using per-request OpenAI API key: {censor_key(openai_api_key)}")
            if 'pinecone_api_key' in request_overrides and request_overrides['pinecone_api_key']:
                pinecone_api_key = str(request_overrides['pinecone_api_key'])
                Actor.log.info(f"Using per-request Pinecone API key: {censor_key(pinecone_api_key)}")
            if 'index_name' in request_overrides and request_overrides['index_name']:
                index_name = str(request_overrides['index_name'])
                Actor.log.info(f"Using per-request index: {index_name}")
            if 'llm_model' in request_overrides and request_overrides['llm_model']:
                llm_model = str(request_overrides['llm_model'])
                Actor.log.info(f"Using per-request LLM model: {llm_model}")

        Actor.log.info(f"Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        Actor.log.info(f"Settings: k={k}, threshold={threshold}, recency_weight={recency_weight}")
        Actor.log.info(f"LLM model: {llm_model}")

        # Initialize services
        embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model=EMBEDDING_MODEL,
        )

        pinecone_client = Pinecone(api_key=pinecone_api_key)
        index = pinecone_client.Index(index_name)

        # Use namespace if configured
        namespace = _config.namespace if _config.namespace else None
        Actor.log.info(f"Using Pinecone index: {index_name}, namespace: {namespace or '(default)'}")

        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace,
        )

        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=llm_model,
            temperature=0.0,
        )

        # Retrieve context with recency-aware ranking
        documents = retrieve_documents(
            vector_store,
            question,
            k,
            threshold,
            recency_weight,
            recency_decay_days,
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
                "start_template": lambda _: start_template,
                "context": lambda _: context,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)
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
                "status": "ready" if _config else "not_configured",
                "message": "Interviews RAG Actor - Standby Mode",
                "running_on_apify": is_running_on_apify(),
                "config": {
                    "openai_key": "configured" if _config else "MISSING",
                    "pinecone_key": "configured" if _config else "MISSING",
                    "index_name": _config.index_name if _config else "MISSING",
                    "llm_model": LLM_MODEL,
                    "k": _config.k if _config else None,
                    "threshold": _config.threshold if _config else None,
                    "recency_weight": _config.recency_weight if _config else None,
                },
                "endpoints": {
                    "POST /query": "Submit a RAG query with 'question' field",
                    "GET /health": "Health check",
                },
                "overridable_params": [
                    "openai_api_key", "pinecone_api_key", "index_name", "llm_model",
                    "k", "threshold", "recency_weight", "recency_decay_days", "start_template"
                ]
            })
            return

        if path == '/query':
            # Handle query via GET with query parameters
            query_params = parse_qs(parsed_url.query)
            input_data = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}

            question = input_data.get('question')
            if not question:
                self.send_json_response(
                    {"error": "Missing 'question' parameter", "error_type": "validation"},
                    status_code=400
                )
                return

            result = process_rag_query(question, input_data)
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

            question = input_data.get('question')
            if not question:
                self.send_json_response(
                    {"error": "Missing 'question' field in request body", "error_type": "validation"},
                    status_code=400
                )
                return

            result = process_rag_query(question, input_data)
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

    Actor.log.info(f"✓ HTTP server started on port {HTTP_PORT}")
    Actor.log.info(f"✓ Endpoints:")
    Actor.log.info(f"    GET  /health  - Health check")
    Actor.log.info(f"    POST /query   - Submit RAG query")
    Actor.log.info("=" * 60)
    Actor.log.info("Waiting for requests...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        Actor.log.info("Shutting down HTTP server...")
        httpd.shutdown()


async def main() -> None:
    """Main entry point for the Apify Actor (Standby mode only)."""
    global _config

    async with Actor:
        Actor.log.info("=" * 60)
        Actor.log.info("Interviews RAG Actor starting (Standby mode)...")
        Actor.log.info("=" * 60)

        # Log environment info
        Actor.log.info(f"Running on: {'APIFY PLATFORM' if is_running_on_apify() else 'LOCAL (.env)'}")
        Actor.log.info(f"HTTP Port: {HTTP_PORT}")

        # Log environment variables (censored)
        Actor.log.info("Configuration from environment:")
        Actor.log.info(f"  OPENAI_API_KEY: {censor_key(os.getenv('OPENAI_API_KEY'))}")
        Actor.log.info(f"  PINECONE_API_KEY: {censor_key(os.getenv('PINECONE_API_KEY'))}")
        Actor.log.info(f"  INDEX_NAME: {os.getenv('INDEX_NAME', '(not set)')}")
        Actor.log.info(f"  K: {os.getenv('K', '(default: 20)')}")
        Actor.log.info(f"  THRESHOLD: {os.getenv('THRESHOLD', '(default: 0.5)')}")
        Actor.log.info(f"  RECENCY_WEIGHT: {os.getenv('RECENCY_WEIGHT', '(default: 0.2)')}")
        Actor.log.info(f"  NAMESPACE: {os.getenv('NAMESPACE', '(default)')}")

        # Load configuration from environment variables
        try:
            _config = load_config()
            Actor.log.info("✓ Configuration loaded successfully")
            Actor.log.info(f"  Index: {_config.index_name}")
            Actor.log.info(f"  k={_config.k}, threshold={_config.threshold}, recency_weight={_config.recency_weight}")
        except ValueError as e:
            Actor.log.error(f"✗ Configuration error: {e}")
            Actor.log.error("Please set the required environment variables:")
            Actor.log.error("  - OPENAI_API_KEY")
            Actor.log.error("  - PINECONE_API_KEY")
            Actor.log.error("  - INDEX_NAME")
            if is_running_on_apify():
                Actor.log.error("Configure these in Actor Settings → Environment Variables")
            else:
                Actor.log.error("Create a .env file in the project root with these variables")
            return

        # Start HTTP server
        Actor.log.info("=" * 60)
        Actor.log.info("Starting HTTP server...")
        Actor.log.info("=" * 60)
        await asyncio.to_thread(run_http_server)
