from __future__ import annotations
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .assembly import Assembly
    from ..brain import Brain


class AssemblySampler(ABC):
    @staticmethod
    @abstractmethod
    def sample_neurons(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain) -> Iterable[int]:
        pass
