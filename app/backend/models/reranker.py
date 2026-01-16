import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging

from models.device import get_device, get_device_for_reranker
from models.reranker_models import RERANKER_MODELS

logger = logging.getLogger(__name__)

class RerankerModel:
    def __init__(
        self,
        model_type = "gte",
        device: Optional[str] = None,
        max_workers: int = 1,
        **model_kwargs
    ):
        self._device = get_device(preferred=device)
        self._model_type = model_type
        
        if model_type not in RERANKER_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(RERANKER_MODELS.keys())}")
        
        model_cls = RERANKER_MODELS[model_type]
        self._model = model_cls(device=self._device, **model_kwargs)
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"RerankerModel initialized: type={model_type}, device={self._device}")
    
    def rerank_sync(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        return self._model.rerank(query, documents, top_k)
    
    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._model.rerank,
            query, documents, top_k
        )

