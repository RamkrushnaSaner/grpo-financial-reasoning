"""Name -> implementation lookup: the one extension mechanism in the package.

Every swappable component (task, reward component, advantage estimator, loss,
model loader) is registered under a short name and selected by that name in a
config file. Adding a variant means writing one function and decorating it;
nothing that calls it has to change.

    from grpo.algorithm import LOSSES

    @LOSSES.register("my_loss")
    def my_loss(logp_new, logp_old, logp_ref, adv, mask, *, epsilon, beta): ...

then set ``algorithm.loss: my_loss`` in a config.
"""
from __future__ import annotations

from typing import Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items and self._items[name] is not obj:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {self.names()}") from None

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items
