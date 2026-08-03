"""Loading configuration from a ``.env`` file.

The project has exactly two environment variables and no configuration framework, so a short
parser is cheaper than a dependency. It is deliberately not a full ``dotenv`` implementation:
no interpolation, no multi-line values, no trailing-comment stripping. It reads what
``.env.example`` documents and nothing more.

Values already present in the environment always win. An explicit ``export`` should beat a
file the user wrote weeks ago and forgot.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["load_env_file"]


def load_env_file(path: str | Path = ".env") -> dict[str, str]:
    """Load ``KEY=value`` pairs from *path* into :data:`os.environ`.

    Returns the variables that were actually set, which excludes any that the environment
    already defined. A missing file is not an error — running without one is the normal case,
    since everything but live generation works with no configuration at all.
    """
    file = Path(path)
    if not file.is_file():
        return {}

    loaded: dict[str, str] = {}
    for line in file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, _, value = stripped.partition("=")
        key = key.strip().removeprefix("export").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]

        # An empty assignment is how `.env.example` marks a variable the user still has to
        # fill in. Setting it would shadow a real value exported in the shell.
        if not key or not value or key in os.environ:
            continue

        os.environ[key] = value
        loaded[key] = value

    return loaded
