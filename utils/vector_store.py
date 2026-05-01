"""
Vector Store for SQL Examples using Nomic Embeddings and FAISS
Can be run directly to build the index: python vector_store.py
"""
import json
import os
import time
import numpy as np
import faiss
import ollama
from pathlib import Path
from typing import List, Dict, Optional

# Cache one client per host so we never create a new SSL context per call.
# The ollama library binds its default client at import time, so OLLAMA_HOST
# set after import has no effect without this patch.
_ollama_client_cache: dict = {}

def _get_ollama_client():
    host = os.environ.get("OLLAMA_HOST", "")
    if not host:
        return None
    if host not in _ollama_client_cache:
        _ollama_client_cache[host] = ollama.Client(host=host, timeout=30)
    return _ollama_client_cache[host]


class SQLVectorStore:
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """Initialize vector store with Nomic embeddings"""
        self.embedding_model = embedding_model
        self.index = None
        self.examples = []
        self.dimension = None

    def _get_embedding(self, text: str, retries: int = 3) -> np.ndarray:
        """Get embedding vector for text using Nomic, with retry on transient errors."""
        for attempt in range(retries):
            try:
                client = _get_ollama_client()
                if client:
                    response = client.embeddings(model=self.embedding_model, prompt=text)
                else:
                    response = ollama.embeddings(model=self.embedding_model, prompt=text)
                return np.array(response['embedding'], dtype=np.float32)
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    print(f"Error getting embedding after {retries} attempts: {e}")
        return None
    
    def build_index_from_spider(
        self, 
        spider_json_path: str, 
        save_path: str
    ) -> bool:
        """Build FAISS index from Spider dataset"""
        try:
            # Load Spider data
            print("📂 Loading Spider dataset...")
            with open(spider_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   Loaded {len(data)} examples")
            
            # Get embeddings for all questions
            total = len(data)
            print(f"🔄 Generating embeddings for {total} examples...")
            embeddings = []
            valid_examples = []
            errors = 0
            embed_start = time.time()

            for i, example in enumerate(data):
                question = example.get('question', '')
                if not question:
                    continue

                embedding = self._get_embedding(question)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_examples.append(example)
                else:
                    errors += 1

                done = i + 1
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - embed_start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    pct = done / total * 100
                    print(
                        f"   [{done:4d}/{total}] {pct:5.1f}%  "
                        f"elapsed {elapsed:6.1f}s  "
                        f"rate {rate:.1f}/s  "
                        f"ETA {eta:5.0f}s  "
                        f"errors {errors}",
                        flush=True
                    )

            if not embeddings:
                print("❌ No valid embeddings generated")
                return False

            total_embed_time = time.time() - embed_start
            print(f"   Done — {len(embeddings)} embeddings in {total_embed_time:.1f}s ({errors} skipped)")

            # Convert to numpy array
            t0 = time.time()
            embeddings_matrix = np.vstack(embeddings)
            self.dimension = embeddings_matrix.shape[1]
            print(f"   Stacked matrix {embeddings_matrix.shape} in {time.time()-t0:.1f}s")

            # Build FAISS index
            print("🔨 Building FAISS index...", flush=True)
            t0 = time.time()
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            self.index.add(embeddings_matrix)
            print(f"   Index built in {time.time()-t0:.1f}s")

            self.examples = valid_examples

            # Save index and examples
            print(f"💾 Saving to {save_path}...", flush=True)
            t0 = time.time()
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            faiss.write_index(self.index, str(save_dir / "faiss.index"))

            with open(save_dir / "examples.json", 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, indent=2)

            metadata = {
                'dimension': self.dimension,
                'num_examples': len(self.examples),
                'embedding_model': self.embedding_model
            }
            with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"   Saved in {time.time()-t0:.1f}s")
            
            print("✅ Index built and saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error building index: {e}")
            return False
    
    def load_index(self, load_path: str) -> bool:
        """Load existing FAISS index"""
        try:
            load_dir = Path(load_path)
            
            print(f"📂 Loading index from {load_path}...")
            
            # Load FAISS index
            self.index = faiss.read_index(str(load_dir / "faiss.index"))
            
            # Load examples
            with open(load_dir / "examples.json", 'r', encoding='utf-8') as f:
                self.examples = json.load(f)
            
            # Load metadata
            with open(load_dir / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                self.dimension = metadata['dimension']
            
            print(f"✅ Loaded {len(self.examples)} examples (dim: {self.dimension})")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def search(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Dict]:
        """Search for similar examples"""
        if self.index is None:
            print("❌ Index not loaded")
            return []
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return []
            
            # Normalize for cosine similarity
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.examples):
                    result = self.examples[idx].copy()
                    result['similarity_score'] = float(distances[0][i])
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {}
        
        databases = set()
        for example in self.examples:
            databases.add(example.get('db_id', 'unknown'))
        
        return {
            'total_vectors': len(self.examples),
            'unique_databases': len(databases),
            'dimension': self.dimension,
            'embedding_model': self.embedding_model
        }


# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    # Build from train_spider.json (not dev/test — using eval data as retrieval source is leakage)
    _script_dir = Path(__file__).parent.parent
    SPIDER_TRAIN_PATH = str(_script_dir / "data" / "spider" / "spider_data" / "train_spider.json")
    SAVE_PATH = "./vector_store"
    EMBEDDING_MODEL = "nomic-embed-text"

    print("🚀 Building FAISS Vector Index for Spider Dataset")
    print("=" * 60)
    print(f"Spider Dataset: {SPIDER_TRAIN_PATH}")
    print(f"Save Location: {SAVE_PATH}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)

    # Initialize and build
    store = SQLVectorStore(embedding_model=EMBEDDING_MODEL)
    success = store.build_index_from_spider(SPIDER_TRAIN_PATH, SAVE_PATH)
    
    if success:
        print("\n✅ Index built successfully!")
        
        # Show statistics
        stats = store.get_statistics()
        print("\n📊 Index Statistics:")
        print(f"  - Total vectors: {stats['total_vectors']}")
        print(f"  - Unique databases: {stats['unique_databases']}")
        print(f"  - Embedding dimension: {stats['dimension']}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        test_queries = [
            "Show me all singers",
            "Find students with high GPA",
            "Count the number of orders"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = store.search(query, k=2)
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['question'][:60]}... (score: {result['similarity_score']:.3f})")
        
        print("\n" + "=" * 60)
        print("✅ All done! You can now use this index in your app.")
        print(f"   Load it using: SQLVectorStore().load_index('{SAVE_PATH}')")
    else:
        print("\n❌ Failed to build index")