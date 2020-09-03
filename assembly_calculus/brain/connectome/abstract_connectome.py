from __future__ import annotations  # import annotations from later version of python.
# We need it here to annadiane that connectome has a method which returns itself

from collections import defaultdict
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, TypeVar, Mapping, Generic, Callable, Any, Set, Iterable, Iterator
import numpy as np

#from wrapt import ObjectProxy  # Needed to pip install

from assembly_calculus.brain.components import BrainPart, Area, Stimulus, Connection


# The wrapt library implements easy to use wrapper objects, which delegates everything to the object you are
# using. It's very convenient to use (it can be used exactly in the same way).
# More info and examples:
# https://wrapt.readthedocs.io/en/latest/wrappers.html

@dataclass(frozen=True)
class NDArraySetMappingProxy(Mapping[Area, Set[int]]):
    proxied: Dict[Area, np.ndarray]

    def __getitem__(self, item: Area) -> Set[int]:
        return set(self.proxied[item])

    def __setitem__(self, key: Area, value: Iterable[int]):
        value = set(value if value is not None else [])
        if len(value) != key.k and len(value) != 0:
            raise ValueError('Invalid number of winners!')
        self.proxied[key] = np.asarray(list(value), dtype=int)

    def __len__(self) -> int:
        return len(self.proxied)

    def __iter__(self) -> Iterator[Area]:
        return iter(self.proxied)


class AbstractConnectome(metaclass=ABCMeta):
    """
    Represent the graph of connections between areas and stimuli of the brain.
    This is a generic abstract class which offer a good infrastructure for building new models of connectome.
    You should implement some of the parts which are left for private case. For example when and how the connectome
    should be initialized, how the connections are represented.

    Attributes:
        areas: List of area objects in the connectome
        stimuli: List of stimulus objects in the connectome
        connections: Dictionary from tuples of BrainPart(Stimulus/Area) and Area to some object which
        represent the connection (e.g. numpy matrix). Each connection is held ObjectProxy which will
        make the connection.
        to be saved by reference. (This makes the get_subconnectome routine much easier to implement)

    """

    def __init__(self, p, areas=None, stimuli=None):
        self.areas: List[Area] = []
        self.stimuli: List[Stimulus] = []
        self._winners: Dict[Area, np.ndarray] = defaultdict(lambda: np.ndarray(shape=(0,), dtype=int))
        self.support: Dict[Area, Set[int]] = defaultdict(lambda: set())
        self.connections: Dict[Tuple[BrainPart, Area], Connection] = {}
        self.p = p
        self._plasticity_disabled = False

        if areas:
            self.areas = areas
        if stimuli:
            self.stimuli = stimuli

    def add_area(self, area: Area):
        self.areas.append(area)

    def add_stimulus(self, stimulus: Stimulus):
        self.stimuli.append(stimulus)

    @property
    def plasticity_disabled(self):
        return self._plasticity_disabled

    @property
    def plasticity(self) -> bool:
        return not self._plasticity_disabled

    @plasticity.setter
    def plasticity(self, mode: bool):
        self._plasticity_disabled = not mode

    @plasticity_disabled.setter
    def plasticity_disabled(self, value):
        self._plasticity_disabled = value

    @property
    def winners(self) -> Mapping[Area, Set[int]]:
        return NDArraySetMappingProxy(self._winners)

    @abstractmethod
    def get_sources(self, area: Area) -> List[BrainPart]:
        """
        Retrieve all parts with connection to specific areas, according to the current connectome
        :param area: area which we need the connections to
        :return: List of all connections to the area
        """
        pass

    def __repr__(self):
        return f'{self.__class__.__name__} with {len(self.areas)} areas, and {len(self.stimuli)} stimuli'

    def fire(self, connections: Dict[BrainPart, List[Area]]):
        """

        :param connections: The connections on which you want to perform the project
        :return:
        """
        pass
