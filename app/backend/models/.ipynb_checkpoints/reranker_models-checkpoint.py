import numpy as np
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)


class GTEReranker:
    def __init__(self, model_id='Alibaba-NLP/gte-multilingual-reranker-base', device="cpu", max_length=2048):
        self.model = CrossEncoder(model_id, max_length=max_length, device=device, trust_remote_code=True)
        logger.info(f"Loaded GTE Reranker on {device}")
    
    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(documents[idx], float(scores[idx])) for idx in ranked_indices]


class MSMarcoReranker:
    def __init__(self, model_id='cross-encoder/ms-marco-MiniLM-L6-v2', device="cpu", max_length=512):
        self.model = CrossEncoder(model_id, max_length=max_length, device=device)
        logger.info(f"Loaded MS-MARCO Reranker on {device}")
    
    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(documents[idx], float(scores[idx])) for idx in ranked_indices]


class JinaReranker:
    def __init__(self, model_id='jinaai/jina-reranker-v2-base-multilingual', device="cpu", max_length=2048):
        self.model = CrossEncoder(
            model_id,
            automodel_args={"torch_dtype": "auto"},
            trust_remote_code=True,
            max_length=max_length
            device=device
        )
        logger.info(f"Loaded Jina Reranker on {device}")
    
    def rerank(self, query, documents, top_k=5):
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, convert_to_tensor=False)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [(documents[idx], float(scores[idx])) for idx in ranked_indices]


RERANKER_MODELS = {
    "gte": GTEReranker,
    "msmarco": MSMarcoReranker,
    "jina": JinaReranker,
}
