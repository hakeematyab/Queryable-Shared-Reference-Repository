import logging
from minicheck.minicheck import MiniCheck

logger = logging.getLogger(__name__)


class MiniCheckDetector:        
    def __init__(
        self,
        model_id: str = 'roberta-large',
        device: str = "cpu",
        enable_prefix_caching: bool = False
    ):
        
        self.device = device
        self.model_id = model_id
        
        self.scorer = MiniCheck(
            model_name=model_id,
            enable_prefix_caching=enable_prefix_caching,
        )
        logger.info(f"Loaded MiniCheck with backbone: {model_id}")
    
    def detect(
        self, 
        claims, 
        sources
    ):        
        argmax, scores, _, _ = self.scorer.score(docs=sources, claims=claims)
        score = float(scores[0])
        
        if self.model_id == "roberta-large":
            if score >= 0.265:
                decision = -1
            elif score >= 0.105:
                decision = 0
            else:
                decision = 1
        else:
            decision = int(argmax[0])
        
        return {"score": score, "decision": decision}


HALLUCINATION_MODELS = {
    "roberta": lambda **kw: MiniCheckDetector(model_id="roberta-large", **kw),
    "deberta": lambda **kw: MiniCheckDetector(model_id="deberta-v3-large", **kw),
    "flan-t5": lambda **kw: MiniCheckDetector(model_id="flan-t5-large", **kw),
}