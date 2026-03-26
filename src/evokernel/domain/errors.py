class EvoKernelError(Exception):
    """Base error for domain and configuration failures."""


class ConfigLoadError(EvoKernelError):
    """Raised when runtime configuration cannot be loaded."""
