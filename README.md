# Interviews RAG Actor

An Apify Actor that performs Retrieval-Augmented Generation (RAG) on meeting notes stored in Pinecone vector store. It retrieves relevant context and generates answers using GPT-4o.

## Features

- **Vector Search**: Retrieves relevant documents from Pinecone using similarity search
- **GPT-4o Integration**: Generates comprehensive answers using OpenAI's GPT-4o model
- **Citation Support**: Includes source citations with Notion URLs and dates
- **Configurable**: Adjustable retrieval parameters (k, threshold)

## Input

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `question` | string | ✅ | Your question about the meeting notes |
| `index_name` | string | ✅ | Pinecone index name containing embeddings |
| `pinecone_key` | string | ✅ | Pinecone API key |
| `openai_key` | string | ✅ | OpenAI API key |
| `k` | integer | ❌ | Number of documents to retrieve (default: 10) |
| `threshold` | number | ❌ | Similarity threshold 0.0-1.0 (default: 0.6) |
| `start_template` | string | ❌ | Custom system prompt |

You can also set credentials via environment variables: `OPENAI_API_KEY`, `PINECONE_API_KEY`, `INDEX_NAME`.

## Output

```json
{
    "answer": "The answer in markdown format...",
    "citations": [
        {
            "source_number": 1,
            "notion_url": "https://notion.so/...",
            "date": "2024-01-15"
        }
    ],
    "sources_used": 5
}
```

## Local Development

```bash
# Run locally
apify run

# Deploy to Apify
apify login
apify push
```

## How It Works

1. Validates input parameters
2. Connects to Pinecone and retrieves relevant documents
3. Formats context with source metadata
4. Generates answer using GPT-4o with citation instructions
5. Extracts and returns citations with the response
