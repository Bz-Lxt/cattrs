from __future__ import annotations

from threading import local
from typing import Any, Callable

from attrs import frozen


@frozen
class AttributeOverride:
    omit_if_default: bool | None = None
    rename: str | None = None
    omit: bool | None = None  # Omit the field completely.
    struct_hook: Callable[[Any, Any], Any] | None = None  # Structure hook to use.
    unstruct_hook: Callable[[Any], Any] | None = None  # Structure hook to use.


neutral = AttributeOverride()
already_generating = local()

#: Attribute set on built-in composite hooks (classes, containers, ...) so
#: the explicit-stack engine in ``cattrs._deep`` can recognize and drive
#: them iteratively. User-registered and leaf hooks never have it set.
ITERATIVE_MARKER = "_cattrs_iterative"
