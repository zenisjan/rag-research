# Interviews RAG Actor

An Apify Actor that performs Retrieval-Augmented Generation (RAG) on meeting notes stored in Pinecone vector store. It retrieves relevant context and generates answers using GPT-4o.

## Features

- **Vector Search**: Retrieves relevant documents from Pinecone using similarity search
- **GPT-4o Integration**: Generates comprehensive answers using OpenAI's GPT-4o model
- **Citation Support**: Includes source citations with Notion URLs and dates
- **Recency-Aware Ranking**: Balances semantic similarity with document freshness
- **Standby Mode**: Runs as an HTTP server for real-time API requests
- **Configurable**: Adjustable retrieval parameters (k, threshold, recency)

## Standby Mode (HTTP API)

When running in Standby mode, the Actor operates as an HTTP server. Configure API keys once at Actor startup, then send questions via HTTP requests.

### Actor Input (Startup Configuration)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `pinecone_key` | string | ✅ | Pinecone API key |
| `openai_key` | string | ✅ | OpenAI API key |
| `index_name` | string | ✅ | Pinecone index name containing embeddings |
| `k` | integer | ❌ | Number of documents to retrieve (default: 10) |
| `threshold` | number | ❌ | Similarity threshold 0.0-1.0 (default: 0.6) |
| `recency_weight` | number | ❌ | Balance between similarity and recency 0.0-1.0 (default: 0.2) |
| `recency_decay_days` | integer | ❌ | Half-life for recency scoring in days (default: 180) |
| `start_template` | string | ❌ | Custom system prompt |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` or `/health` | Health check and configuration status |
| `POST` | `/` or `/query` | Submit a RAG query |
| `GET` | `/query?question=...` | Submit query via URL parameters |

### POST Request

**Endpoint:** `POST /query` or `POST /`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**

```json
{
    "question": "What did customers say about pricing?"
}
```

**Optional parameters** (override startup config):

```json
{
    "question": "What feedback did we get on the new feature?",
    "k": 5,
    "threshold": 0.7,
    "recency_weight": 0.3,
    "start_template": "Answer concisely based on the context."
}
```

**Example with cURL:**

```bash
curl -X POST https://your-actor.apify.actor/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main customer pain points?"}'
```

### Response

```json
{
    "answer": "Based on the meeting notes, customers mentioned several pain points...",
    "citations": [
        {
            "source_number": 1,
            "notion_url": "https://notion.so/...",
            "date": "2025-01-02"
        }
    ],
    "sources_used": 5
}
```

### Error Response

```json
{
    "error": "Question is required",
    "error_type": "validation"
}
```

## Batch Mode

For one-off queries, provide all parameters in the Actor input including `question`.

### Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | ✅ | Your question about the meeting notes |
| `index_name` | string | ✅ | Pinecone index name containing embeddings |
| `pinecone_key` | string | ✅ | Pinecone API key |
| `openai_key` | string | ✅ | OpenAI API key |
| `k` | integer | ❌ | Number of documents to retrieve (default: 10) |
| `threshold` | number | ❌ | Similarity threshold 0.0-1.0 (default: 0.6) |
| `recency_weight` | number | ❌ | Balance between similarity and recency (default: 0.2) |
| `recency_decay_days` | integer | ❌ | Half-life for recency scoring (default: 180) |
| `start_template` | string | ❌ | Custom system prompt |

You can also set credentials via environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `INDEX_NAME`.

## Local Development

```bash
# Run locally (batch mode)
apify run

# Run locally in Standby mode
ACTOR_STANDBY_PORT=8080 apify run

# Deploy to Apify
apify login
apify push
```

## How It Works

1. Validates input parameters and loads configuration
2. Connects to Pinecone and retrieves relevant documents
3. Applies recency-aware ranking to prioritize fresh content
4. Formats context with source metadata
5. Generates answer using GPT-4o with citation instructions
6. Extracts and returns citations with the response
