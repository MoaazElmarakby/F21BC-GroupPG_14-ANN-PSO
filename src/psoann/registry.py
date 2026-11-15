"""A tiny named registry shared by every extension point (leaf module: imports nothing from psoann)."""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps a string name (as used in configs and overrides) to an implementation."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T, *, overwrite: bool = False) -> T:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind} name must be a non-empty string")
        if name in self._items and not overwrite:
            raise ValueError(f"{self.kind} {name!r} is already registered (pass overwrite=True to replace it)")
        self._items[name] = item
        return item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ValueError(
                f"unknown {self.kind} {name!r}; available: {', '.join(self.names()) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
