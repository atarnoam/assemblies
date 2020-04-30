""" Configurable brain assembly model for simulations and research.
Author: Daniel Mitropolsky, 2018

This module contains classes to represent different elements of a brain simulation:
    - Area - Represents an individual area of the brain, with the relevant parameters.
    - Connectomes are the connections between neurons. They have weights, which are initialized randomly but
        due to plasticity they can updated every time some neuron fires. These weights are represented by numpy arrays.
        The ones that are not random, because they were influenced by previous projections, are referred to as the 'support'.
    - Winners in a given 'round' are the specific neurons that fired in that round.
        In any specific area, these will be the 'k' neurons with the highest value flown into them.
        These are also the only neurons whose connectome weights get updated.
    - Stimulus - Represents a random stimulus that can be applied to any part of the brain.
        When a stimulus is created it is initialized randomly, but when applied multiple times this will change.
        This is equivalent to k neurons from an unknown part of the brain firing and their (initially, random)
        connectomes decide how this stimulus affects a given area of the brain.
    - Brain - A class representing a simulated brain, with it's different areas, stimulus, and all the connectome weights.
        A brain is initialized as a random graph, and it is maintained in a 'sparse' representation,
        meaning that all neurons that have their original, random connectome weights (0 or 1) are not saved explicitly.
    - Assembly - TODO define and express in code
"""
from typing import List, Mapping, Dict
from collections import defaultdict
from numpy.core._multiarray_umath import ndarray


class Stimulus:
    """ Represents a random stimulus that can be applied to any part of the brain.
    That is, a specific set of k neurons that fire together that do not reside in
    any of the brain areas. These k neurons can be though of as representing a
    specific input stimulus.
    A stimulus can be connected to any areas and each pair of stimulus neuron and
    an area neuron initially have a synapse with probability p (Brain.p).

    The data for the synaptic weights is found in the downstream areas (the areas
    that this stimulus goes into).

    Attributes:
        k: number of neurons that fire
    """

    def __init__(self, k: int):
        self.k = k


class Area:
    """Represents an individual area of the brain.

    The list of neurons that are firing is given by 'winners'. It is updated through application of 'Brain.project',
    where the set of '_new_winners' is calculated and only updated once all brain areas settle on their new winners.
    The winners are the 'k' neurons with the highest value going in each round.

    Initially, most computation are represented implicitly. Winners are represented explicitly, and the changes in
    their incoming synapses are maintained in 'Brain.connectomes' and 'Brain.stimuli_connectomes'. The explicit neurons
    are represented by indices starting with 0 up to 'support_size'-1.

    Since it is initialized randomly, all the programmer needs to provide for initialization is the number 'n' of neurons,
    number 'k' of winners in any given round (meaning the k neurons with heights values will fire),
    and the parameter 'beta' of plasticity controlling connectome weight updates.

    TODO: remove '_new_winners'.
    TODO: remove 'name'. We prefer to use variable names to refer to areas.

    Attributes:
        n: number of neurons in this brain area
        k: number of winners in each round
        beta: plasticity parameter for self-connections
        stimulus_beta: plasticity parameters for connections from each incoming stimulus
        area_beta: plasticity parameters for connections from each incoming area
        support_size: The number of neurons that are represented explicitly (= total number of previous winners)
        winners: List of current winners. That is, 'k' top neurons from previous round.
        _new_support_size: the size of the support for the new update. Should be 'support_size' + 'num_first_winners'.
        _new_winners: During the projection process, a new set of winners is formed. The winners are only
            updated when the projection ends, so that the newly computed winners won't affect computation
        num_first_winners: should be equal to 'len(_new_winners)'
    """

    def __init__(self, name: str, n: int, k: int, beta: float = 0.05):
        self.name = name
        self.n = n
        self.k = k
        self.beta = beta
        self.stimulus_beta: Dict[str, float] = {}
        self.area_beta: Dict[str, float] = {}
        self.support = [0] * self.n
        self.support_size: int = 0
        self.winners: List[int] = []
        self._new_support_size: int = 0
        self._new_winners: List[int] = []
        self.num_first_winners: int = -1

    def update_winners(self) -> None:
        """ This function updates the list of winners for this area after a projection step.

            TODO: redesign this so that the list of new winners is not saved in area.
        """
        self.winners = self._new_winners
        self.support_size = self._new_support_size


class Brain:
    """ Represents an abstract brain type.

    TODO:   The implementation should enable the creation of a brain with default choice of underlying implementation
            simply and implicitly unless the user is interested in that.
    TODO:   Make it easy to change plasticity rules. Specifically, I'm interested in adding the following:
            1) Homeostasis - after each iteration reduce all weights in some process. Say w<-max(1, w/(1+beta'))
            2) Having negative weights that strengthen whenever something (perhaps for 'almost-winners')
            3) Capping weights.
                Say, only update weights when w<10.
                Say, update multiplicatively by (1+beta)(10-w)/9+(w-1)/9. (that will also converge to 10)


    Attributes:
    areas: A mapping from area names to Area objects representing them.
    stimuli: A mapping from stimulus names to Stimulus objects representing them.
    stimuli_connectomes: Maps each pair of (stimulus,area) to the ndarray representing the synaptic weights among
        stimulus neurons and neurons in the support of area.
    connectomes: Maps each pair of areas to the ndarray representing the synaptic weights among neurons in
        the support.
    p: Probability of connectome (edge) existing between two neurons (vertices)
    """
    def __init__(self, p: float):
        self.areas: Dict[str, Area] = {}
        self.stimuli: Dict[str, Stimulus] = {}
        self.stimuli_connectomes: Dict[str, Dict[str, ndarray]] = {}
        self.connectomes: Dict[str, Dict[str, ndarray]] = {}
        self.p: float = p

    def add_stimulus(self, name: str, k: int) -> None:
        pass

    def add_area(self, name: str, n: int, k: int, beta: float) -> None:
        pass

    def project(self, stim_to_area: Mapping[str, List[str]],
                area_to_area: Mapping[str, List[str]]) -> None:
        """ Project is the basic operation where some stimuli and some areas are activated,
        with only specified connections between them active.

        :param stim_to_area: Dictionary that matches to each stimuli applied a list of areas to project into.
            Example: {"stim1":["A"], "stim2":["C","A"]}
        :param area_to_area: Dictionary that matches for each area a list of areas to project into.
            Note that an area can also be projected into itself.
            Example: {"A":["A","B"],"C":["C","A"]}
        """
        stim_in: defaultdict[str, List[str]] = defaultdict(lambda: [])
        area_in: defaultdict[str, List[str]] = defaultdict(lambda: [])

        # Validate stim_area, area_area well defined
        # Set stim_in to be the Dictionary that matches for every area the list of input stimuli.
        # Set areas_in to be the Dictionary that matches for every area the list of input areas.
        for stim, areas in stim_to_area.items():
            if stim not in self.stimuli:
                raise IndexError(stim + " not in brain.stimuli")
            for area in areas:
                if area not in self.areas:
                    raise IndexError(area + " not in brain.areas")
                stim_in[area].append(stim)
        for from_area, to_areas in area_to_area.items():
            if from_area not in self.areas:
                raise IndexError(from_area + " not in brain.areas")
            for to_area in to_areas:
                if to_area not in self.areas:
                    raise IndexError(to_area + " not in brain.areas")
                area_in[to_area].append(from_area)

        # to_update is the set of all areas that receive input
        to_update = set().union(list(stim_in.keys()), list(area_in.keys()))

        for area in to_update:
            num_first_winners = self.project_into(self.areas[area], stim_in[area], area_in[area])
            self.areas[area].num_first_winners = num_first_winners

        # once done everything, for each area in to_update: area.update_winners()
        for area in to_update:
            self.areas[area].update_winners()

    def project_into(self, area: Area, from_stimuli: List[str], from_areas: List[str]) -> int:
        return 0
