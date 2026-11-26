class ServiceError(ValueError):
    """An actionable failure whose message can be sent to the caller."""

    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
