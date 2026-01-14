import pytest
from unittest.mock import MagicMock
from retrieval.retrieval import HybridSearcher


class TestHybridSearcher:
    def test_init(self):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_reranker = MagicMock()
        
        searcher = HybridSearcher(mock_store, mock_embedder, mock_reranker)
        
        assert searcher is not None
        assert searcher.store is mock_store
        assert searcher.embedder is mock_embedder
        assert searcher.reranker is mock_reranker
    
    def test_init_without_reranker(self):
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        
        searcher = HybridSearcher(mock_store, mock_embedder, reranker=None)
        
        assert searcher is not None
        assert searcher.reranker is None
