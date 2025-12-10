import os
from typing import Optional
import logging
import torch

logger = logging.getLogger(__name__)

class DeviceType:
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def get_device(
    preferred: Optional[str] = None,
    enable_mps: bool = True,
    enable_mps_fallback: bool = True
) -> str:
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            logger.info("Using CUDA (explicitly requested)")
            return DeviceType.CUDA
        elif preferred == "mps" and torch.backends.mps.is_available():
            if enable_mps_fallback:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            logger.info("Using MPS (explicitly requested)")
            return DeviceType.MPS
        elif preferred == "cpu":
            logger.info("Using CPU (explicitly requested)")
            return DeviceType.CPU
        else:
            logger.warning(f"Requested device '{preferred}' not available, auto-detecting...")
    
    if torch.cuda.is_available():
        logger.info(f"Using CUDA - {torch.cuda.get_device_name(0)}")
        return DeviceType.CUDA
    
    if enable_mps and torch.backends.mps.is_available():
        if enable_mps_fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        logger.info("Using MPS (Apple Silicon)")
        return DeviceType.MPS
    
    logger.info("Using CPU")
    return DeviceType.CPU


def get_device_for_reranker(preferred: Optional[str] = None) -> str:    
    if preferred:
        return get_device(preferred=preferred)
    
    if torch.cuda.is_available():
        return DeviceType.CUDA
    
    logger.info("Using CPU for reranker (MPS has known compatibility issues)")
    return DeviceType.CPU