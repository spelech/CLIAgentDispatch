class DispatchError(Exception):
    """Base exception for all CLIAgentDispatch errors."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ExecutorNotFoundError(DispatchError):
    """Raised when the specified executor binary or server cannot be found."""

    pass


class ExecutorTimeoutError(DispatchError):
    """Raised when an executor command exceeds the allowed timeout."""

    pass


class ExecutionFailedError(DispatchError):
    """Raised when an executor exits with a non-zero exit status or error payload."""

    pass


class InvalidWorkspaceError(DispatchError):
    """Raised when a specified workspace directory does not exist or is inaccessible."""

    pass
