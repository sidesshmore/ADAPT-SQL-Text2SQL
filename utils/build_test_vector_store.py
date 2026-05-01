import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.vector_store import SQLVectorStore

store = SQLVectorStore()
store.build_index_from_spider(
    str(Path(__file__).parent.parent / "data" / "spider" / "spider_data" / "test.json"),
    str(Path(__file__).parent.parent / "vector_store_test")
)
