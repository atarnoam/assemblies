from numpy.core import ndarray
from typing import Dict, List, Iterable
import numpy as np
from collections import defaultdict

from assembly_calculus.brain.components import Area, BrainPart, Stimulus, Connection
from assembly_calculus.brain.connectome.abstract_connectome import AbstractConnectome
from assembly_calculus.brain.performance import RandomMatrix


class Connectome(AbstractConnectome):
    """
    Implementation of a random based connectome, based on the abstract connectome.
    The object representing the connection in here is ndarray from numpy
    """
    def __init__(self, p: float, areas=None, stimuli=None, initialize=False):
        """
        :param p: The attribute p for the probability of an edge to exits
        :param areas: list of areas
        :param stimuli: list of stimuli
        :param initialize: Whether or not to initialize the connectome of the brain.
        """
        super(Connectome, self).__init__(p, areas, stimuli)

        self.rng = RandomMatrix()

        if initialize:
            self._initialize_parts((areas or []) + (stimuli or []))
    # TODO: check what to do with plasticity

    def add_area(self, area: Area):
        super().add_area(area)
        self._initialize_parts([area])

    def add_stimulus(self, stimulus: Stimulus):
        super().add_stimulus(stimulus)
        self._initialize_parts([stimulus])

    def _initialize_parts(self, parts: List[BrainPart]) -> None:
        """
        Initialize all the connections to and from the given brain parts.
        :param parts: List of stimuli and areas to initialize
        """
        for part in parts:
            for other in self.areas + self.stimuli:
                self._initialize_connection(part, other)
                if isinstance(part, Area) and part != other:
                    self._initialize_connection(other, part)

    def _initialize_connection(self, part: BrainPart, area: Area) -> None:
        """
        Initalize the connection from brain part to an area
        :param part: Stimulus or Area which the connection should come from
        :param area: Area which the connection go to
        """
        synapses = self.rng.multi_generate(area.n, part.n, self.p).reshape((part.n, area.n), order='F')
        self.connections[part, area] = Connection(part, area, synapses)

    def get_sources(self, area: Area) -> List[BrainPart]:
        """ Get sources to area """
        return [source for source, dest in self.connections if dest == area]

    def _update_connection(self, source: BrainPart, area: Area, new_winners: Dict[Area, ndarray]) -> None:
        """
        Update one connection (based on the plasticity).
        A helper function for update_connectomes.
        :param source: the source the area
        :param area: the area to _update the connection
        :param new_winners: the new winners per area
        """
        connection = self.connections[source, area]
        beta = connection.beta
        source_neurons: Iterable[int] = \
            range(source.n) if isinstance(source, Stimulus) else self._winners[source]
        # Note that this uses numpy vectorization to multiply a whole matrix by a scalar.
        connection.synapses[source_neurons, new_winners[area][:, None]] *= (1 + beta)

    def update_connectomes(self, new_winners: Dict[Area, ndarray], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the connectomes of the areas with new winners, based on the plasticity.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        if self._plasticity_disabled:
            return
        for area in new_winners:
            for source in sources[area]:
                self._update_connection(source, area, new_winners)

    def update_winners(self, new_winners: Dict[Area, ndarray], sources: Dict[Area, List[BrainPart]]) -> None:
        """
        Update the winners of areas with new winners.
        :param new_winners: the new winners per area
        :param sources: the sources of each area
        """
        to_update = sources.keys()
        for area in to_update:
            self._winners[area] = new_winners[area]
            self.support[area].update(new_winners[area])

    def _fire_into(self, area: Area, sources: List[BrainPart]) -> np.ndarray:
        """
        Fire multiple stimuli and area assemblies into area 'area' at the same time.
        :param area: The area projected into
        :param sources: List of separate brain parts whose assemblies we will projected into this area
        :return: Returns new winners of the area
        """
        # Calculate the total input for each neuron from other given areas' winners and given stimuli.
        # Said total inputs list is saved in prev_winner_inputs
        src_areas = [src for src in sources if isinstance(src, Area)]
        src_stimuli = [src for src in sources if isinstance(src, Stimulus)]
        for part in sources:
            if (part, area) not in self.connections:
                self._initialize_connection(part, area)

        prev_winner_inputs: ndarray = np.zeros(area.n)
        for source in src_areas:
            area_connectome = self.connections[source, area]
            prev_winner_inputs += sum((area_connectome.synapses[winner, :] for winner in self._winners[source]))
        if src_stimuli:
            prev_winner_inputs += sum(self.connections[stim, area].synapses.sum(axis=0) for stim in src_stimuli)
        return np.argpartition(prev_winner_inputs, area.n - area.k)[-area.k:]

    def fire(self, connections: Dict[BrainPart, List[Area]], *, override_winners: Dict[Area, List[int]] = None,
             enable_plasticity=True):
        """
        Fire is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.
        :param connections: A dictionary of connections to use in the projection, for example {area1
        """

        sources_mapping: defaultdict[Area, List[BrainPart]] = defaultdict(lambda: [])

        for part, areas in connections.items():
            for area in areas:
                sources_mapping[area] = sources_mapping[area] or []
                sources_mapping[area].append(part)

        # to_update is the set of all areas that receive input
        to_update = sources_mapping.keys()

        new_winners: Dict[Area, ndarray] = dict()
        for area in to_update:
            if override_winners and area in override_winners:  # In case of enforcing some winners to some area
                new_winners[area] = np.array(override_winners[area])
            else:
                new_winners[area] = self._fire_into(area, sources_mapping[area])

        if enable_plasticity:
            self.update_connectomes(new_winners, sources_mapping)

        self.update_winners(new_winners, sources_mapping)
