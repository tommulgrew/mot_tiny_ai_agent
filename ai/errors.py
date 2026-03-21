class AIClientError(Exception):
    """General AI chat completion client error"""

class AIClientTokenOverflowError(AIClientError):
    """Raised when token context overflows and the client cannot recover"""

