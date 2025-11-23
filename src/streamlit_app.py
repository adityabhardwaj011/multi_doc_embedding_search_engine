import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessor import DocumentPreprocessor
from src.embedder import EmbeddingGenerator
from src.search_engine import VectorSearchEngine


@st.cache_resource
def load_search_engine():
    preprocessor = DocumentPreprocessor()
    embedder = EmbeddingGenerator()
    search_engine = VectorSearchEngine(embedder, preprocessor)
    
    if search_engine.load_index():
        documents = preprocessor.load_all_documents()
        for doc in documents:
            search_engine.documents[doc["doc_id"]] = doc
            search_engine.doc_ids.append(doc["doc_id"])
        return search_engine, len(documents)
    else:
        return None, 0


def main():
    st.set_page_config(
        page_title="Document Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Multi-document Embedding Search Engine")
    st.markdown("Search through documents using semantic similarity")
    
    search_engine, doc_count = load_search_engine()
    
    if search_engine is None or doc_count == 0:
        st.error("⚠️ Search engine not initialized!")
        st.info("""
        Please follow these steps:
        1. Download the dataset: `python download_dataset.py`
        2. Generate embeddings: `python -m src.generate_embeddings`
        3. Refresh this page
        """)
        return
    
    st.sidebar.header("Search Configuration")
    st.sidebar.info(f"📚 **{doc_count}** documents indexed")
    
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., quantum physics basics",
        key="search_query"
    )
    
    top_k = st.sidebar.slider("Number of results:", 1, 20, 5)
    
    if st.button("🔍 Search", type="primary") or query:
        if not query.strip():
            st.warning("Please enter a search query")
            return
        
        with st.spinner("Searching..."):
            try:
                results = search_engine.search(query, top_k=top_k)
                
                if not results:
                    st.info("No results found. Try a different query.")
                    return
                
                st.success(f"Found {len(results)} results")
                
                for i, result in enumerate(results, 1):
                    with st.expander(
                        f"📄 Result {i}: {result['filename']} (Score: {result['score']:.3f})",
                        expanded=(i == 1)
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**Preview:**")
                            st.text(result['preview'])
                        
                        with col2:
                            st.markdown("**Metadata:**")
                            st.write(f"**Doc ID:** {result['doc_id']}")
                            st.write(f"**Length:** {result['doc_length']:,} chars")
                            st.write(f"**Score:** {result['score']:.3f}")
                        
                        st.markdown("---")
                        st.markdown("**Explanation:**")
                        explanation = result['explanation']
                        st.write(explanation['why_matched'])
                        
                        if explanation['overlapping_keywords']:
                            st.markdown("**Overlapping Keywords:**")
                            keywords = ", ".join(explanation['overlapping_keywords'])
                            st.code(keywords)
                        
                        st.metric("Overlap Ratio", f"{explanation['overlap_ratio']:.1%}")
                
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
    
    with st.sidebar.expander("Cache Statistics"):
        from src.cache_manager import CacheManager
        cache_manager = CacheManager()
        stats = cache_manager.get_cache_stats()
        st.write(f"**Cached embeddings:** {stats['total_cached']}")
        if stats['oldest_entry']:
            st.write(f"**Oldest entry:** {stats['oldest_entry'][:10]}")
        if stats['newest_entry']:
            st.write(f"**Newest entry:** {stats['newest_entry'][:10]}")


if __name__ == "__main__":
    main()

