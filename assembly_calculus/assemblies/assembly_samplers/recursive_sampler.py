from __future__ import annotations
from assembly_calculus.assemblies.assembly_sampler import AssemblySampler
from assembly_calculus.utils.brain_utils import fire_many
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from assembly_calculus.assemblies.assembly import Assembly
    from assembly_calculus.brain import Brain


# TODO: update documentation (`read` method, etc). Explain what is the meaning of "read driver" and what is the purpose
# of this class structure in general.
class RecursiveSampler(AssemblySampler):
    """
    A class representing a reader that obtains information about an assembly using the 'read' method.
    The method works by recursively firing areas from the top of the parent tree of the assembly,
    and examining which neurons were fired.
    Note: This is the default read driver.
    """

    @staticmethod
    def sample_neurons(assembly: Assembly, preserve_brain: bool = False, *, brain: Brain):
        """
        Read the winners from given assembly in given brain recursively using fire_many
        and return the result.
        :param assembly: the assembly object
        :param preserve_brain: a boolean representing whether we want to change the brain state or not
        :param brain: the brain object
        :return: the winners as read from the area that we've fired up
        """
        with brain.freeze(freeze=preserve_brain):
            fire_many(brain, assembly.parents, assembly.area)
            return brain.winners[assembly.area]