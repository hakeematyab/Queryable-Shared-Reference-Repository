import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union
import logging

from models.device import get_device
from models.hallucination_models import HALLUCINATION_MODELS

logger = logging.getLogger(__name__)


class HallucinationDetector:
    def __init__(
        self,
        model_type = "roberta",
        device: Optional[str] = None,
        max_workers: int = 1,
        **model_kwargs
    ):
        self._device = get_device(preferred=device)
        self._model_type = model_type
        
        if model_type not in HALLUCINATION_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(HALLUCINATION_MODELS.keys())}")
        
        model_factory = HALLUCINATION_MODELS[model_type]
        self._model = model_factory(device=self._device, **model_kwargs)
        
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info(f"HallucinationDetector initialized: type={model_type}")
    
    
    def detect_sync(
        self, 
        claims: Union[str, list[str]], 
        sources: Union[str, list[str]]
    ) -> list[float]:
        return self._model.detect(claims, sources)
    
    
    async def detect(
        self, 
        claims: Union[str, list[str]], 
        sources: Union[str, list[str]]
    ) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._model.detect,
            claims, sources
        )