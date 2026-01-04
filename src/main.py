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
from typing import List, Dict, Any, Tuple, Optional
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


# Configuration constants
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o"
DEFAULT_K = 10
DEFAULT_THRESHOLD = 0.6
DEFAULT_RECENCY_WEIGHT = 0.2
DEFAULT_RECENCY_DECAY_DAYS = 180
HTTP_PORT = int(os.getenv('ACTOR_STANDBY_PORT', '8080'))

# Global store for resolved credentials (from env vars or Actor input)
# These are loaded once at Actor startup and used for all requests
_credentials: Dict[str, Any] = {}
_config_loaded: bool = False


def is_running_on_apify() -> bool:
    """Check if we're running on Apify platform (not local development)."""
    # APIFY_IS_AT_HOME is set to "1" when running on Apify platform
    return os.getenv('APIFY_IS_AT_HOME') == '1'


def censor_key(key: Optional[str]) -> str:
    """Censor an API key for safe logging."""
    if not key:
        return "(not set)"
    if len(key) < 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def log_config_source(name: str, value: Optional[str], source: str) -> None:
    """Log where a config value came from."""
    status = "✓" if value else "✗"
    censored = censor_key(value) if "key" in name.lower() else (value or "(not set)")
    Actor.log.info(f"  {status} {name}: {censored} (from {source})")


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


def get_value_with_source(
    input_data: dict,
    *keys: str,
    env_var: Optional[str] = None,
    default: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Get value from env or input data with fallback.
    Returns tuple of (value, source) for logging.
    
    Priority on Apify platform:
        1. Environment variables (configured in Actor settings)
        2. Cached credentials (from startup)
        3. Request input data
        4. Default
    
    Priority for local development:
        1. Actor input (INPUT.json)
        2. Environment variables
        3. Request input data
        4. Default
    """
    on_apify = is_running_on_apify()
    
    if on_apify:
        # On Apify: env vars first (configured in Actor settings)
        if env_var:
            val = os.getenv(env_var)
            if val and val.strip():
                return val.strip(), f"env:{env_var}"
        
        # Then cached credentials (loaded at startup)
        for key in keys:
            val = _credentials.get(key)
            if val and str(val).strip():
                return str(val).strip(), f"cached:{key}"
        
        # Then request input data
        for key in keys:
            val = input_data.get(key)
            if val and str(val).strip():
                return str(val).strip(), f"input:{key}"
    else:
        # Local development: Actor input first
        for key in keys:
            val = input_data.get(key)
            if val and str(val).strip():
                return str(val).strip(), f"input:{key}"
        
        # Then check cached credentials
        for key in keys:
            val = _credentials.get(key)
            if val and str(val).strip():
                return str(val).strip(), f"cached:{key}"
        
        # Then env vars
        if env_var:
            val = os.getenv(env_var)
            if val and val.strip():
                return val.strip(), f"env:{env_var}"
    
    # Return default
    if default is not None:
        return default, "default"
    return None, "not_found"


def get_config(input_data: dict, require_question: bool = True) -> Config:
    """
    Validate inputs and create Config object.
    Environment variables take priority over input data.
    
    Args:
        input_data: Dictionary with configuration values
        require_question: If False, allows missing question (for startup validation)
    """
    openai_api_key, openai_src = get_value_with_source(
        input_data, 'openai_key', 'openai_api_key', env_var='OPENAI_API_KEY'
    )
    pinecone_api_key, pinecone_src = get_value_with_source(
        input_data, 'pinecone_key', 'pinecone_api_key', env_var='PINECONE_API_KEY'
    )
    index_name, index_src = get_value_with_source(
        input_data, 'index_name', 'index', env_var='INDEX_NAME'
    )
    question, question_src = get_value_with_source(
        input_data, 'question', 'query', env_var='QUESTION'
    )

    # Parse numeric values
    k_raw, _ = get_value_with_source(input_data, 'k', env_var='K', default=str(DEFAULT_K))
    try:
        k = max(1, min(50, int(k_raw))) if k_raw else DEFAULT_K
    except (ValueError, TypeError):
        k = DEFAULT_K

    threshold_raw, _ = get_value_with_source(
        input_data, 'threshold', env_var='THRESHOLD', default=str(DEFAULT_THRESHOLD)
    )
    try:
        threshold = max(0.0, min(1.0, float(threshold_raw))) if threshold_raw else DEFAULT_THRESHOLD
    except (ValueError, TypeError):
        threshold = DEFAULT_THRESHOLD

    recency_weight_raw, _ = get_value_with_source(
        input_data, 'recency_weight', env_var='RECENCY_WEIGHT', default=str(DEFAULT_RECENCY_WEIGHT)
    )
    try:
        recency_weight = max(0.0, min(1.0, float(recency_weight_raw))) if recency_weight_raw else DEFAULT_RECENCY_WEIGHT
    except (ValueError, TypeError):
        recency_weight = DEFAULT_RECENCY_WEIGHT

    recency_decay_raw, _ = get_value_with_source(
        input_data, 'recency_decay_days', env_var='RECENCY_DECAY_DAYS', default=str(DEFAULT_RECENCY_DECAY_DAYS)
    )
    try:
        recency_decay_days = max(1, int(recency_decay_raw)) if recency_decay_raw else DEFAULT_RECENCY_DECAY_DAYS
    except (ValueError, TypeError):
        recency_decay_days = DEFAULT_RECENCY_DECAY_DAYS

    start_template, _ = get_value_with_source(
        input_data,
        'start_template',
        env_var='START_TEMPLATE',
        default='Answer the question based on the context provided.'
    )

    # Validate required fields
    missing = []
    if not openai_api_key:
        missing.append("OpenAI API key (env: OPENAI_API_KEY or input: openai_key)")
    if not pinecone_api_key:
        missing.append("Pinecone API key (env: PINECONE_API_KEY or input: pinecone_key)")
    if not index_name:
        missing.append("Index name (env: INDEX_NAME or input: index_name)")
    if require_question and not question:
        missing.append("Question (input: question)")
    
    if missing:
        raise ValueError(f"Missing required configuration: {', '.join(missing)}")

    return Config(
        openai_api_key=openai_api_key,
        pinecone_api_key=pinecone_api_key,
        index_name=index_name,
        question=question or "",  # Empty if not required
        k=k,
        threshold=threshold,
        start_template=start_template or 'Answer the question based on the context provided.',
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


def get_merged_input(request_data: dict) -> dict:
    """
    Merge cached credentials with request-specific data.
    Request data is merged but credentials come from cached or env vars.
    """
    merged = {**_credentials}
    merged.update(request_data)
    return merged


def process_rag_query(input_data: dict) -> dict:
    """
    Process a RAG query and return the result.
    This is the core logic extracted for reuse in both batch and HTTP modes.
    """
    try:
        Actor.log.info("-" * 40)
        Actor.log.info("Processing RAG query...")
        
        # Log request data
        request_keys = list(input_data.keys()) if input_data else []
        Actor.log.debug(f"Request data keys: {request_keys}")
        
        # Merge with cached Actor config (API keys, index, etc.)
        merged_input = get_merged_input(input_data)
        
        Actor.log.debug(f"Merged config keys: {list(merged_input.keys())}")
        
        # Get config - this properly checks env vars AND input data
        # Don't check question manually here, let get_config handle it
        config = get_config(merged_input, require_question=True)
        
        Actor.log.info(f"Question: {config.question[:100]}{'...' if len(config.question) > 100 else ''}")
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
            # Get config status from resolved credentials
            openai_key, _ = get_value_with_source({}, 'openai_key', 'openai_api_key', env_var='OPENAI_API_KEY')
            pinecone_key, _ = get_value_with_source({}, 'pinecone_key', 'pinecone_api_key', env_var='PINECONE_API_KEY')
            index_name, _ = get_value_with_source({}, 'index_name', 'index', env_var='INDEX_NAME')
            
            self.send_json_response({
                "status": "ready" if _config_loaded else "initializing",
                "message": "Interviews RAG Actor is running in Standby mode",
                "config_loaded": _config_loaded,
                "running_on_apify": is_running_on_apify(),
                "config": {
                    "openai_key": "configured" if openai_key else "MISSING",
                    "pinecone_key": "configured" if pinecone_key else "MISSING",
                    "index_name": index_name or "MISSING",
                },
                "endpoints": {
                    "POST /query": "Submit a RAG query (only 'question' required)",
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
                Actor.log.info(f"POST body raw: {body}")
                input_data = json.loads(body.decode('utf-8'))
                Actor.log.info(f"POST parsed input_data: {input_data}")
                Actor.log.info(f"POST question key present: {'question' in input_data}")
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
    
    Actor.log.info(f"✓ HTTP server started on port {HTTP_PORT}")
    Actor.log.info(f"✓ Endpoints available:")
    Actor.log.info(f"    GET  /        - Health check / status")
    Actor.log.info(f"    GET  /health  - Health check")
    Actor.log.info(f"    POST /query   - Submit RAG query")
    Actor.log.info(f"    POST /        - Submit RAG query (alias)")
    Actor.log.info("=" * 60)
    Actor.log.info("Waiting for requests...")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        Actor.log.info("Shutting down HTTP server...")
        httpd.shutdown()


async def run_batch_mode() -> None:
    """Run in traditional batch mode (process single input and exit)."""
    # In batch mode, _actor_config is already loaded, just process it
    result = process_rag_query({})
    await Actor.push_data(result)


async def load_actor_config() -> Dict[str, Any]:
    """
    Load and cache credentials from environment variables and Actor input.
    
    On Apify platform:
        - Credentials should be configured as Actor environment variables
        - Actor.get_input() is used for optional overrides and other settings
    
    Local development:
        - Credentials come from INPUT.json (Actor input)
        - Environment variables as fallback
    
    Returns the loaded configuration dict.
    """
    global _credentials, _config_loaded
    
    on_apify = is_running_on_apify()
    
    Actor.log.info("=" * 60)
    Actor.log.info("Loading Actor configuration...")
    Actor.log.info(f"Running on: {'APIFY PLATFORM' if on_apify else 'LOCAL DEVELOPMENT'}")
    Actor.log.info("=" * 60)
    
    # Load from Actor input (key-value store)
    actor_input = await Actor.get_input() or {}
    
    if actor_input:
        Actor.log.info(f"Actor.get_input() returned {len(actor_input)} keys: {list(actor_input.keys())}")
    else:
        Actor.log.info("Actor.get_input() returned empty (expected in Standby mode)")
    
    # Build credentials dict - this is what we cache for all requests
    _credentials = {}
    
    if on_apify:
        # On Apify: Load from environment variables (primary source)
        Actor.log.info("Loading credentials from environment variables...")
        
        if os.getenv('OPENAI_API_KEY'):
            _credentials['openai_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('PINECONE_API_KEY'):
            _credentials['pinecone_key'] = os.getenv('PINECONE_API_KEY')
        if os.getenv('INDEX_NAME'):
            _credentials['index_name'] = os.getenv('INDEX_NAME')
        
        # Also load optional settings from env vars
        if os.getenv('K'):
            _credentials['k'] = os.getenv('K')
        if os.getenv('THRESHOLD'):
            _credentials['threshold'] = os.getenv('THRESHOLD')
        if os.getenv('RECENCY_WEIGHT'):
            _credentials['recency_weight'] = os.getenv('RECENCY_WEIGHT')
        if os.getenv('RECENCY_DECAY_DAYS'):
            _credentials['recency_decay_days'] = os.getenv('RECENCY_DECAY_DAYS')
        if os.getenv('START_TEMPLATE'):
            _credentials['start_template'] = os.getenv('START_TEMPLATE')
        
        # Merge any Actor input (lower priority, for overrides)
        for key, value in actor_input.items():
            if key not in _credentials and value:
                _credentials[key] = value
    else:
        # Local development: Load from Actor input (INPUT.json)
        Actor.log.info("Loading credentials from Actor input (INPUT.json)...")
        _credentials = dict(actor_input)
    
    # Log configuration sources
    Actor.log.info("-" * 40)
    Actor.log.info("Configuration resolution:")
    
    # Check each key source (pass empty dict since we want to see where values come from)
    openai_key, openai_src = get_value_with_source({}, 'openai_key', 'openai_api_key', env_var='OPENAI_API_KEY')
    pinecone_key, pinecone_src = get_value_with_source({}, 'pinecone_key', 'pinecone_api_key', env_var='PINECONE_API_KEY')
    index_name, index_src = get_value_with_source({}, 'index_name', 'index', env_var='INDEX_NAME')
    
    log_config_source("openai_key", openai_key, openai_src)
    log_config_source("pinecone_key", pinecone_key, pinecone_src)
    log_config_source("index_name", index_name, index_src)
    
    # Log optional config
    k_val, _ = get_value_with_source({}, 'k', env_var='K', default=str(DEFAULT_K))
    threshold_val, _ = get_value_with_source({}, 'threshold', env_var='THRESHOLD', default=str(DEFAULT_THRESHOLD))
    recency_weight_val, _ = get_value_with_source({}, 'recency_weight', env_var='RECENCY_WEIGHT', default=str(DEFAULT_RECENCY_WEIGHT))
    
    Actor.log.info(f"  ✓ k: {k_val}, threshold: {threshold_val}, recency_weight: {recency_weight_val}")
    Actor.log.info("-" * 40)
    
    _config_loaded = True
    return _credentials


def validate_startup_config() -> None:
    """
    Validate that required configuration is present at startup.
    Raises ValueError if critical config is missing.
    """
    Actor.log.info("Validating startup configuration...")
    
    try:
        # Validate without requiring question (that comes from requests)
        config = get_config(_credentials, require_question=False)
        Actor.log.info(f"✓ Configuration valid - ready to serve requests")
        Actor.log.info(f"  Index: {config.index_name}")
        Actor.log.info(f"  OpenAI key: {censor_key(config.openai_api_key)}")
        Actor.log.info(f"  Pinecone key: {censor_key(config.pinecone_api_key)}")
        return config
    except ValueError as e:
        Actor.log.error(f"✗ Configuration validation failed: {e}")
        raise


async def main() -> None:
    """Main entry point for the Apify Actor."""
    async with Actor:
        Actor.log.info("=" * 60)
        Actor.log.info("Interviews RAG Actor starting...")
        Actor.log.info("=" * 60)
        
        # Log environment info
        is_standby = os.getenv('ACTOR_STANDBY_PORT') is not None
        Actor.log.info(f"Mode: {'STANDBY' if is_standby else 'BATCH'}")
        Actor.log.info(f"ACTOR_STANDBY_PORT: {os.getenv('ACTOR_STANDBY_PORT', '(not set)')}")
        Actor.log.info(f"HTTP Port: {HTTP_PORT}")
        
        # Log relevant env vars (censored)
        Actor.log.info("Environment variables check:")
        Actor.log.info(f"  OPENAI_API_KEY: {censor_key(os.getenv('OPENAI_API_KEY'))}")
        Actor.log.info(f"  PINECONE_API_KEY: {censor_key(os.getenv('PINECONE_API_KEY'))}")
        Actor.log.info(f"  INDEX_NAME: {os.getenv('INDEX_NAME', '(not set)')}")
        
        # Load and cache Actor input (API keys, index name, etc.)
        await load_actor_config()
        
        # Validate configuration at startup (fail fast if missing required config)
        try:
            validate_startup_config()
        except ValueError as e:
            Actor.log.error(f"Failed to start: {e}")
            if is_running_on_apify():
                Actor.log.error("Please configure OPENAI_API_KEY, PINECONE_API_KEY, and INDEX_NAME as Actor environment variables")
            else:
                Actor.log.error("Please configure openai_key, pinecone_key, and index_name in INPUT.json")
            return

        if is_standby:
            Actor.log.info("=" * 60)
            Actor.log.info("Starting HTTP server for Standby mode...")
            Actor.log.info("=" * 60)
            # Run HTTP server in a thread to not block the async loop
            await asyncio.to_thread(run_http_server)
        else:
            Actor.log.info("=" * 60)
            Actor.log.info("Running in batch mode...")
            Actor.log.info("=" * 60)
            await run_batch_mode()
