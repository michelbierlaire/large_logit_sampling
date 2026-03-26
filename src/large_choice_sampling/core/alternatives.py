"""Core abstraction for alternatives in large choice sets.

This module defines a minimal, generic representation of a complete
alternative. It is designed to be extended in application-specific
contexts (e.g., route choice, activity scheduling), while enforcing
only one essential requirement: alternatives must be hashable so they
can be stored in sets and dictionaries.

The class is intentionally lightweight and can be enriched with
additional attributes depending on the application.

Michel Bierlaire
Wed Mar 25 2026, 17:13:02
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True)
class Alternative:
    """Base class representing a complete alternative.

    This class provides a minimal structure for alternatives used in
    large choice-set sampling algorithms. It is defined as an immutable
    dataclass to ensure safe usage as dictionary keys or set elements.

    Concrete applications are expected to extend this class with
    additional attributes describing the alternative (e.g., path,
    schedule, attributes).

    :param identifier: A hashable object uniquely identifying the
        alternative. This is the only required attribute.
    """

    identifier: Hashable
