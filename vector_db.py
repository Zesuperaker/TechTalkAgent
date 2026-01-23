import chromadb
import os
from openai import OpenAI

# Initialize Chroma client and OpenRouter client
client = None
collection = None
openrouter_client = None


def init_chroma():
    """Initialize Chroma database with OpenRouter embeddings"""
    global client, collection, openrouter_client

    # Set up Chroma with persistent storage using new API
    client = chromadb.PersistentClient(path="./chroma_data")

    # Initialize OpenRouter client for embeddings
    openrouter_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name="knowledge_base"
    )

    print("Chroma database initialized successfully")


def get_collection():
    """Get the active Chroma collection"""
    global collection
    return collection


def get_embedding(text: str):
    """Get embedding from OpenRouter"""
    global openrouter_client

    response = openrouter_client.embeddings.create(
        model="openai/text-embedding-3-large",
        input=text
    )

    return response.data[0].embedding


def add_documents(documents: list[str], ids: list[str] = None):
    """Add documents to the collection"""
    global collection

    if ids is None:
        ids = [f"doc_{i}" for i in range(len(documents))]

    # Generate embeddings using OpenRouter
    embeddings = [get_embedding(doc) for doc in documents]

    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents
    )

    return len(documents)


def query_documents(query_text: str, n_results: int = 3):
    """Query similar documents from the collection"""
    global collection

    # Generate embedding for query using OpenRouter
    query_embedding = get_embedding(query_text)

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


def get_collection_stats():
    """Get statistics about the collection"""
    global collection
    try:
        count = collection.count()
        return {"count": count}
    except:
        return {"count": 0}