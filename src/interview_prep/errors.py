class BudgetExceeded(RuntimeError):
    """A shared limit prevents further work."""

class UnknownOperation(RuntimeError):
    """An external request may have completed; deliberate retry is required."""

class ProviderFailure(RuntimeError):
    """Safe public provider/configuration failure."""
