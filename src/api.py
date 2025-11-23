from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from .preprocessor import DocumentPreprocessor
from .embedder import EmbeddingGenerator
from .search_engine import VectorSearchEngine


app = FastAPI(
    title="Multi-document Embedding Search Engine",
    description="A lightweight embedding-based search engine with caching",
    version="1.0.0"
)


preprocessor = None
embedder = None
search_engine = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    doc_id: str
    score: float
    preview: str
    explanation: dict
    filename: str
    doc_length: int


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int


@app.on_event("startup")
async def startup_event():
    # Initializing all components when the API starts up
    global preprocessor, embedder, search_engine
    
    print("Initializing search engine components...")
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    search_engine = VectorSearchEngine(embedder, preprocessor)
    
    # Trying to load existing index if available
    if not search_engine.load_index():
        print("No existing index found. Please run embedding generation first.")
    else:
        # Loading document metadata so we can show previews and explanations
        documents = preprocessor.load_all_documents()
        for doc in documents:
            search_engine.documents[doc["doc_id"]] = doc
            search_engine.doc_ids.append(doc["doc_id"])
        print(f"Loaded {len(documents)} documents into search engine")


@app.get("/")
async def root():
    return {
        "message": "Multi-document Embedding Search Engine API",
        "endpoints": {
            "/search": "POST - Search documents",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "index_loaded": search_engine.index is not None if search_engine else False,
        "documents_indexed": len(search_engine.doc_ids) if search_engine else 0
    }


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    # Making sure the search engine is ready
    if search_engine is None or search_engine.index is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please generate embeddings first."
        )
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Performing the search
        results = search_engine.search(request.query, top_k=request.top_k)
        
        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            query=request.query,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

