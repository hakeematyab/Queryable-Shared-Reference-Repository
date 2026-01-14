import pytest
import tempfile
import shutil
from pathlib import Path
from retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def temp_dirs():
    chroma_dir = tempfile.mkdtemp()
    bm25_path = tempfile.mktemp(suffix=".pkl")
    yield chroma_dir, bm25_path
    shutil.rmtree(chroma_dir, ignore_errors=True)
    Path(bm25_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def vector_store(temp_dirs):
    chroma_dir, bm25_path = temp_dirs
    return VectorStore(chroma_dir=chroma_dir, bm25_path=bm25_path, collection_name="test_papers")


class TestVectorStore:
    def test_init(self, vector_store):
        assert vector_store is not None
        assert hasattr(vector_store, "chroma")
        assert hasattr(vector_store, "bm25")
    
    def test_load(self, vector_store):
        vector_store.load()
    
    def test_count_returns_dict(self, vector_store):
        result = vector_store.count()
        assert isinstance(result, dict)
        assert "chroma" in result
        assert "bm25" in result
        assert isinstance(result["chroma"], int)
        assert isinstance(result["bm25"], int)
    
    def test_add_and_count(self, vector_store):
        initial_counts = vector_store.count()
        
        ids = ["test_id_1", "test_id_2"]
        texts = ["Test document one.", "Test document two."]
        embeddings = [[0.1] * 768, [0.2] * 768]
        metas = [{"doc_name": "test1.pdf"}, {"doc_name": "test2.pdf"}]
        
        vector_store.add(ids, texts, embeddings, metas)
        
        new_counts = vector_store.count()
        assert new_counts["chroma"] == initial_counts["chroma"] + 2
        assert new_counts["bm25"] == initial_counts["bm25"] + 2
    
    def test_save(self, vector_store):
        vector_store.save()
