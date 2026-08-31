"""Domain exceptions exposed by Frame Quorum."""


class FrameQuorumError(Exception):
    """Base exception for expected, user-actionable failures."""


class ScanError(FrameQuorumError):
    """Raised when an input sequence cannot be scanned."""


class ConfigurationError(FrameQuorumError):
    """Raised when a configuration has contradictory or invalid values."""


class OutputError(FrameQuorumError):
    """Raised when a report bundle cannot be written consistently."""
