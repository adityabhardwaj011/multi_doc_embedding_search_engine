import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import VectorSearchEngine


def main():
    print("=" * 60)
    print("Embedding Generation and Index Building")
    print("=" * 60)
    
    # Setting up all the components we need
    print("\n1. Initializing components...")
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    search_engine = VectorSearchEngine(embedder, preprocessor)
    
    # Loading all text files from the data directory
    print("\n2. Loading documents...")
    documents = preprocessor.load_all_documents()
    
    if not documents:
        print("ERROR: No documents found in data/ directory.")
        print("Please download the dataset first using:")
        print("  python download_dataset.py")
        return
    
    print(f"   Loaded {len(documents)} documents")
    
    # Generate embeddings - will use cache if documents haven't changed
    print("\n3. Generating embeddings (using cache when available)...")
    embeddings = embedder.generate_embeddings_batch(documents, use_cache=True)
    
    print(f"   Generated {len(embeddings)} embeddings")
    
    # Building the FAISS search index so we can find similar documents quickly
    print("\n4. Building FAISS search index...")
    search_engine.build_index(documents, embeddings)
    
    print("\n" + "=" * 60)
    print("SUCCESS: Embeddings generated and index built!")
    print("=" * 60)
    print("\nwe can now start the API server with by either running api.py or streamlit_app.py:")



if __name__ == "__main__":
    main()

