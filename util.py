import json
import logging

def log_dump(obj) -> str:
    """
    Serialize for logging. 
    Not guaranteed to be deserializable again, but more tolerant than a regular
    json.dumps()
    """
    return json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)

def create_logger(name:str, filepath: str, propagate: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filepath)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = propagate
    return logger
