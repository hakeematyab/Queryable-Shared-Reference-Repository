import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("./data/logs")

def setup_session_logging(user_id: str = "server"):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{user_id}_{timestamp}.log"
    
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    
    logging.getLogger().addHandler(handler)
    return log_path
