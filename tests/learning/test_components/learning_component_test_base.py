from unittest import TestCase

from assembly_calculus.learning.components.input import InputStimuli
from assembly_calculus.learning.components.sequence import LearningSequence
from tests.brain_test_utils import BrainTestUtils


class LearningComponentTestBase(TestCase):

    def setUp(self) -> None:
        utils = BrainTestUtils(lazy=False)
        self.brain = utils.create_brain(number_of_areas=3, number_of_stimuli=4,
                                        area_size=100, winners_size=10, add_output_area=True)

        self.sequence = LearningSequence(self.brain)
        self.sequence.add_iteration(input_bits_to_areas={0: ['A'], 1: ['B']})
        self.sequence.add_iteration(areas_to_areas={'A': ['C'], 'B': ['C']})
        self.sequence.add_iteration(areas_to_areas={'C': ['output']})

        self.input_stimuli = InputStimuli(self.brain, 10, 'A', 'B', verbose=False)

