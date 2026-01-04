# Interviews RAG Actor

An Apify Actor that performs Retrieval-Augmented Generation (RAG) on meeting notes stored in Pinecone vector store. It retrieves relevant context and generates answers using GPT-4o.

**This Actor runs only in Standby mode** as an HTTP server for real-time API requests.

## Features

- **Vector Search**: Retrieves relevant documents from Pinecone using similarity search
- **GPT-4o Integration**: Generates comprehensive answers using OpenAI's GPT-4o model
- **Citation Support**: Includes source citations with Notion URLs and dates
- **Recency-Aware Ranking**: Balances semantic similarity with document freshness
- **Standby Mode**: Runs as an HTTP server for real-time API requests
- **Configurable**: Adjustable retrieval parameters (k, threshold, recency)

## Configuration

All configuration is loaded from **environment variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API key for embeddings and LLM |
| `PINECONE_API_KEY` | ✅ | Pinecone API key |
| `INDEX_NAME` | ✅ | Pinecone index name containing embeddings |
| `K` | ❌ | Number of documents to retrieve (default: 20) |
| `THRESHOLD` | ❌ | Similarity threshold 0.0-1.0 (default: 0.3) |
| `RECENCY_WEIGHT` | ❌ | Balance between similarity and recency 0.0-1.0 (default: 0.2) |
| `RECENCY_DECAY_DAYS` | ❌ | Half-life for recency scoring in days (default: 180) |
| `START_TEMPLATE` | ❌ | Custom system prompt |

### Local Development

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-openai-api-key
PINECONE_API_KEY=your-pinecone-api-key
INDEX_NAME=interviews

# Optional
K=20
THRESHOLD=0.3
RECENCY_WEIGHT=0.2
```

### Apify Deployment

Configure environment variables in **Actor Settings → Environment Variables** on Apify Console.

## API Endpoints

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

**Optional per-request overrides:**

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
    "error": "Missing 'question' field in request body",
    "error_type": "validation"
}
```

## Local Development

```bash
# Create .env file with your credentials first

# Run locally in Standby mode
ACTOR_STANDBY_PORT=8080 apify run

# Deploy to Apify
apify login
apify push
```

## How It Works

1. Loads configuration from environment variables at startup
2. Starts HTTP server listening for requests
3. For each query:
   - Connects to Pinecone and retrieves relevant documents
   - Applies recency-aware ranking to prioritize fresh content
   - Formats context with source metadata
   - Generates answer using GPT-4o with citation instructions
   - Extracts and returns citations with the response
