import json
import logging
import random
from numpy import dot, linalg

def log_dump(obj) -> str:
    """
    Serialize for logging. 
    Not guaranteed to be deserializable again, but more tolerant than a regular
    json.dumps()
    """
    return json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)

def create_logger(name:str, filepath: str, level: int | None = None, propagate: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filepath)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = propagate
    if level:
        logger.level = level
        handler.level = level
    return logger

def bellcurverandom(count: int) -> float:
    return sum(random.random() for _ in range(count)) / count

def cosine_similarity(a: list[float], b: list[float]) -> float:
    return dot(a, b) / (linalg.norm(a) * linalg.norm(b))
