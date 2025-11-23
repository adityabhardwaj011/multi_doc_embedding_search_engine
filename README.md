# Multi-document Embedding Search Engine with Caching

A lightweight embedding-based search engine that enables semantic search over 100-200 text documents with efficient caching, vector search using FAISS, and a clean REST API.

## Features

- ✅ **Efficient Embedding Generation**: Uses `sentence-transformers/all-MiniLM-L6-v2` for fast, high-quality embeddings
- ✅ **Local Caching**: SQLite-based caching system prevents recomputing embeddings
- ✅ **Vector Search**: FAISS-powered similarity search with cosine similarity
- ✅ **REST API**: FastAPI-based API with `/search` endpoint
- ✅ **Result Ranking & Explanation**: Detailed explanations for why documents matched
- ✅ **Streamlit UI**: Interactive web interface for searching (bonus feature)
- ✅ **Modular Architecture**: Clean, well-structured codebase

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Document loading and text cleaning
│   ├── cache_manager.py     # SQLite-based embedding cache
│   ├── embedder.py          # Embedding generation with caching
│   ├── search_engine.py     # FAISS vector search with explanations
│   ├── api.py               # FastAPI REST API
│   ├── generate_embeddings.py # Script to generate embeddings
│   └── streamlit_app.py     # Streamlit UI (bonus)
├── data/                    # Text documents (gitignored)
├── cache/                   # Cache files (gitignored)
│   ├── embeddings_cache.db  # SQLite cache database
│   └── faiss_index.index   # FAISS index file
├── download_dataset.py      # Dataset download script
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create and activate virtual environment** (if not already done):
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## How Caching Works

The caching system uses SQLite to store document embeddings with the following structure:

```python
{
    "doc_id": "doc_001",
    "embedding": [...],  # JSON-serialized numpy array
    "hash": "sha256_of_text",
    "updated_at": "timestamp",
    "filename": "doc_001.txt",
    "doc_length": 1234
}
```

### Cache Lookup Process

1. **Hash Computation**: Each document's text is hashed using SHA256
2. **Cache Check**: When generating embeddings, the system checks if:
   - A cached embedding exists for the document ID
   - The cached hash matches the current document hash
3. **Cache Hit**: If hash matches → reuse cached embedding (no recomputation)
4. **Cache Miss**: If hash differs or no cache exists → generate new embedding and store it

### Benefits

- **Speed**: Avoids recomputing embeddings for unchanged documents
- **Efficiency**: Only processes new or modified documents
- **Persistence**: Cache survives between runs
- **Automatic**: No manual cache management needed

## Usage

### Step 1: Download Dataset

Download the 20 Newsgroups dataset (or use your own text files):

```bash
python download_dataset.py
```

This will:
- Download ~11,000 documents from the 20 Newsgroups dataset
- Save them as `doc_001.txt`, `doc_002.txt`, etc. in the `data/` directory
- Remove headers, footers, and quotes for cleaner text

**Alternative**: Place your own `.txt` files in the `data/` directory.

### Step 2: Generate Embeddings

Generate embeddings for all documents and build the search index:

```bash
python -m src.generate_embeddings
```

This script will:
1. Load all documents from `data/` directory
2. Check cache for existing embeddings
3. Generate embeddings only for new/unchanged documents
4. Build FAISS index for fast similarity search
5. Save index to `cache/faiss_index.index`

**Note**: On first run, this may take several minutes. Subsequent runs will be much faster due to caching.

### Step 3: Start the API Server

Start the FastAPI server:

```bash
python -m src.api
```

Or using uvicorn directly:

```bash
uvicorn src.api:app --reload
```

The API will be available at `http://localhost:8000`

### Step 4: Use the API

#### Search Endpoint

**POST** `/search`

Request body:
```json
{
    "query": "quantum physics basics",
    "top_k": 5
}
```

Response:
```json
{
    "results": [
        {
            "doc_id": "doc_014",
            "score": 0.88,
            "preview": "Quantum theory is concerned with...",
            "explanation": {
                "why_matched": "This document matched because it contains keywords: quantum, physics, theory. Similarity score: 0.880, Keyword overlap: 60.0%",
                "overlapping_keywords": ["quantum", "physics", "theory"],
                "overlap_ratio": 0.6,
                "length_normalization": 0.95,
                "similarity_score": 0.88
            },
            "filename": "doc_014.txt",
            "doc_length": 1234
        }
    ],
    "query": "quantum physics basics",
    "total_results": 5
}
```

#### Other Endpoints

- **GET** `/` - API information
- **GET** `/health` - Health check
- **GET** `/docs` - Interactive API documentation (Swagger UI)

### Step 5: Use Streamlit UI (Bonus)

For an interactive web interface:

```bash
streamlit run src/streamlit_app.py
```

This opens a browser-based UI where you can:
- Enter search queries
- View results with explanations
- See cache statistics
- Adjust number of results

## Design Choices

### 1. Embedding Model
- **Choice**: `sentence-transformers/all-MiniLM-L6-v2`
- **Reason**: Fast, efficient, good quality for semantic search, runs locally (no API calls)

### 2. Caching System
- **Choice**: SQLite database
- **Reason**: 
  - Lightweight, no external dependencies
  - Persistent storage
  - Fast lookups with indexed queries
  - Easy to inspect and debug

### 3. Vector Search
- **Choice**: FAISS IndexFlatIP (Inner Product)
- **Reason**:
  - Fast similarity search
  - Normalized embeddings → Inner Product = Cosine Similarity
  - Supports exact search (no approximation needed for this dataset size)
  - Can be saved/loaded for persistence

### 4. API Framework
- **Choice**: FastAPI
- **Reason**:
  - Modern, fast, async support
  - Automatic API documentation
  - Type validation with Pydantic
  - Easy to extend

### 5. Ranking Explanation
- **Implementation**: 
  - Keyword overlap analysis
  - Overlap ratio calculation
  - Document length normalization
  - Semantic similarity score
- **Reason**: Provides transparency and helps users understand why documents matched

## Module Details

### `preprocessor.py`
- Loads `.txt` files from `data/` directory
- Cleans text: lowercase, remove HTML, normalize whitespace
- Computes SHA256 hash for cache lookup
- Extracts metadata: filename, length, hash

### `cache_manager.py`
- SQLite-based cache storage
- Stores embeddings with document hash
- Validates cache hits using hash comparison
- Provides cache statistics

### `embedder.py`
- Wraps sentence-transformers model
- Integrates with cache manager
- Supports batch embedding generation
- Handles query embedding generation

### `search_engine.py`
- Builds and manages FAISS index
- Performs vector similarity search
- Generates ranking explanations
- Handles index persistence

### `api.py`
- FastAPI application
- `/search` endpoint with request/response models
- Health check endpoint
- Automatic API documentation

## Performance Considerations

- **First Run**: May take 5-10 minutes for 100-200 documents (embedding generation)
- **Subsequent Runs**: Near-instant (uses cache)
- **Search Speed**: <100ms for queries (FAISS is very fast)
- **Memory**: ~50-100MB for 200 documents (depends on document length)

## Troubleshooting

### No documents found
- Ensure `.txt` files are in the `data/` directory
- Run `python download_dataset.py` to download sample data

### Index not found
- Run `python -m src.generate_embeddings` first
- Check that `cache/faiss_index.index` exists

### Cache issues
- Clear cache: Delete `cache/embeddings_cache.db`
- Regenerate embeddings: Run `python -m src.generate_embeddings`

### Import errors
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`
- Check Python path includes project root

## Future Enhancements (Not Implemented)

- Query expansion using WordNet or embedding similarity
- Batch embedding with multiprocessing
- Evaluation metrics (precision@k, recall@k)
- Support for other embedding models
- Advanced FAISS indices (IVF, HNSW) for larger datasets

## License

This project is created as part of an assignment for CodeAtRandom AI.

## Author

AI Engineer Intern Assignment - CodeAtRandom AI

