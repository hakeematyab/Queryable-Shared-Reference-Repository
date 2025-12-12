import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import logging

from models.device import get_device
from models.embedding_models import EMBEDDING_MODELS

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(
        self,
        model_type = "gemma",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_workers: int = 1,
        **model_kwargs
    ):
        self._device = get_device(preferred=device)
        self._model_type = model_type
        
        self._model = EMBEDDING_MODELS[model_type](
            device=self._device, batch_size=batch_size, **model_kwargs
        )
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
    
    
    def embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)
    
    def embed_query_sync(self, text: str) -> list[float]:
        return self._model.embed_query(text)
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._model.embed_documents, texts)
    
    async def embed_query(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._model.embed_query, text)
