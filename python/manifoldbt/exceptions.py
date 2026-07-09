"""Exception hierarchy for manifoldbt."""


class BacktesterError(Exception):
    """Base exception for all manifoldbt errors."""


class DataError(BacktesterError):
    """Raised when data loading, versioning, or format issues occur."""


class StrategyError(BacktesterError):
    """Raised when strategy compilation or validation fails."""


class ConfigError(BacktesterError):
    """Raised when backtest configuration is invalid."""


class LicenseError(BacktesterError):
    """Raised when a Pro feature is used without a valid license."""

    def _render_traceback_(self):
        # Jupyter/IPython uses this hook (when present) to render an exception,
        # replacing the default traceback: a Community user hitting a Pro gate
        # sees a short, frame-free notice instead of an internal traceback (file
        # paths, the raise site, etc.). Plain `.py` scripts still get the normal
        # traceback.
        #
        # Bold + the theme's default foreground (no fixed colour): orange washes
        # out on Jupyter's pink error background, the default fg stays readable
        # on any theme. Split across two indented lines with blank lines around
        # so the notice breathes instead of reading as a cramped, clipped strip.
        head, _, tail = str(self).partition(". ")
        lines = ["", f"  \033[1m{head}\033[0m"]
        if tail:
            lines.append(f"  {tail}")
        lines.append("")
        return lines
