import json

def log_dump(obj) -> str:
    """
    Serialize for logging. 
    Not guaranteed to be deserializable again, but more tolerant than a regular
    json.dumps()
    """
    return json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
