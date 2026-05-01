import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.vector_store import SQLVectorStore

# Use train_spider.json as retrieval source for both dev and test evaluation.
# Do NOT use dev.json or test.json — that is data leakage.
ROOT = Path(__file__).parent.parent
store = SQLVectorStore()
store.build_index_from_spider(
    str(ROOT / "data" / "spider" / "spider_data" / "train_spider.json"),
    str(ROOT / "vector_store")
)
