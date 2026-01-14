import pytest
from models.embedding import EmbeddingModel
from models.reranker import RerankerModel
from models.hallucination import HallucinationDetector


class TestEmbeddingModel:
    @pytest.fixture(scope="class")
    def embedding_model(self):
        return EmbeddingModel()
    
    def test_init(self, embedding_model):
        assert embedding_model is not None
        assert hasattr(embedding_model, "_model")
        assert hasattr(embedding_model, "_device")
    
    def test_embed_documents_sync_returns_list(self, embedding_model):
        texts = ["This is a test document.", "Another test document."]
        result = embedding_model.embed_documents_sync(texts)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(emb, list) for emb in result)
        assert all(isinstance(val, float) for emb in result for val in emb)
    
    def test_embed_documents_sync_dimensions_consistent(self, embedding_model):
        texts = ["Short text.", "A slightly longer text for testing."]
        result = embedding_model.embed_documents_sync(texts)
        dims = [len(emb) for emb in result]
        assert dims[0] == dims[1]
        assert dims[0] > 0
    
    def test_embed_documents_sync_empty_list(self, embedding_model):
        result = embedding_model.embed_documents_sync([])
        assert isinstance(result, list)
        assert len(result) == 0


class TestRerankerModel:
    def test_init(self):
        model = RerankerModel()
        assert model is not None
        assert hasattr(model, "_model")
        assert hasattr(model, "_device")
    
    def test_init_invalid_model_type(self):
        with pytest.raises(ValueError):
            RerankerModel(model_type="nonexistent_model")


class TestHallucinationDetector:
    def test_init(self):
        detector = HallucinationDetector()
        assert detector is not None
        assert hasattr(detector, "_model")
        assert hasattr(detector, "_device")
    
    def test_init_invalid_model_type(self):
        with pytest.raises(ValueError):
            HallucinationDetector(model_type="nonexistent_model")
