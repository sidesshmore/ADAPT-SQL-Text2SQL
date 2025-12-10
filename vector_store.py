"""
Vector Store for SQL Examples using Nomic Embeddings and FAISS
Can be run directly to build the index: python vector_store.py
"""
import json
import numpy as np
import faiss
import ollama
from pathlib import Path
from typing import List, Dict, Optional


class SQLVectorStore:
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        """Initialize vector store with Nomic embeddings"""
        self.embedding_model = embedding_model
        self.index = None
        self.examples = []
        self.dimension = None
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using Nomic"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            embedding = np.array(response['embedding'], dtype=np.float32)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
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
            print("🔄 Generating embeddings...")
            embeddings = []
            valid_examples = []
            
            for i, example in enumerate(data):
                if (i + 1) % 100 == 0:
                    print(f"   Progress: {i + 1}/{len(data)}")
                
                question = example.get('question', '')
                if not question:
                    continue
                
                embedding = self._get_embedding(question)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_examples.append(example)
            
            if not embeddings:
                print("❌ No valid embeddings generated")
                return False
            
            # Convert to numpy array
            embeddings_matrix = np.vstack(embeddings)
            self.dimension = embeddings_matrix.shape[1]
            
            print(f"   Generated {len(embeddings)} embeddings (dim: {self.dimension})")
            
            # Build FAISS index
            print("🔨 Building FAISS index...")
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            self.index.add(embeddings_matrix)
            
            self.examples = valid_examples
            
            # Save index and examples
            print(f"💾 Saving to {save_path}...")
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(save_dir / "faiss.index"))
            
            # Save examples
            with open(save_dir / "examples.json", 'w', encoding='utf-8') as f:
                json.dump(self.examples, f, indent=2)
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'num_examples': len(self.examples),
                'embedding_model': self.embedding_model
            }
            with open(save_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
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
    # Configuration
    SPIDER_DEV_PATH = "/Users/sidessh/ADAPT-SQL/data/spider/dev.json"
    SAVE_PATH = "./vector_store"
    EMBEDDING_MODEL = "nomic-embed-text"
    
    print("🚀 Building FAISS Vector Index for Spider Dataset")
    print("=" * 60)
    print(f"Spider Dataset: {SPIDER_DEV_PATH}")
    print(f"Save Location: {SAVE_PATH}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)
    
    # Initialize and build
    store = SQLVectorStore(embedding_model=EMBEDDING_MODEL)
    success = store.build_index_from_spider(SPIDER_DEV_PATH, SAVE_PATH)
    
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