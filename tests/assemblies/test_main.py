import gc
from os import environ

from assembly_calculus import Connectome, bake
from assembly_calculus.utils import overlap, protecc_ram
from assembly_calculus.utils.brain_utils import _construct_firing_order

CERTAINTY_REPEAT = 25
EFFECTIVE_REPEAT = 3

if environ.get('PROTECC_MY_RAM', False):
    protecc_ram(0.90)


# TODO4: remove code duplication inside tests. reuse code using methods. test code should be treated as regular code :)
# Response: Not code duplication in my opinion, this is used in totally different context and 2 lines is fine,
#           moving to function will be confusing.

def test_projection(recipe, assembly_a, assembly_b, area_c):
    with recipe:
        assembly_ac = assembly_a >> area_c
        assembly_bc = assembly_b >> area_c

    for _ in range(CERTAINTY_REPEAT):
        with bake(recipe, 0.1, Connectome, train_repeat=100, effective_repeat=EFFECTIVE_REPEAT) as brain:
            assembly_a >> area_c
            assert area_c.active_assembly == assembly_ac, "Separate assemblies have merged :("

            assembly_b >> area_c
            assert area_c.active_assembly == assembly_bc, "Separate assemblies have merged :("

        gc.collect()


def test_associate(recipe, assembly_a, assembly_b, area_c):
    def average_winners(assembly, times):
        winners = set()
        for _ in range(times):
            winners.update(set(assembly.sample_neurons(preserve_brain=False)))
        return winners

    with recipe:
        assembly_ac = assembly_a >> area_c
        assembly_bc = assembly_b >> area_c

    for _ in range(CERTAINTY_REPEAT):
        with bake(recipe, 0.1, Connectome, train_repeat=10, effective_repeat=EFFECTIVE_REPEAT) as brain:
            winners_a = average_winners(assembly_ac, 5)
            winners_b = average_winners(assembly_bc, 5)
            assert overlap(winners_a, winners_b) <= 0.25, "Assemblies have associated without associate"

        gc.collect()

    with recipe:
        (assembly_ac | ...).associate(assembly_bc | ...)

    for _ in range(CERTAINTY_REPEAT):
        with bake(recipe, 0.1, Connectome, train_repeat=10, effective_repeat=EFFECTIVE_REPEAT) as brain:
            winners_a = average_winners(assembly_ac, 5)
            winners_b = average_winners(assembly_bc, 5)
            assert overlap(winners_a, winners_b) > 0.25, "Assemblies haven't associated"

        gc.collect()


def test_merge(recipe, assembly_a, assembly_b, area_b, area_c):
    for _ in range(CERTAINTY_REPEAT):
        with bake(recipe, 0.1, Connectome, train_repeat=10, effective_repeat=EFFECTIVE_REPEAT) as brain:
            assembly_a >> area_c
            brain.next_round(subconnectome={area_c: [area_b]}, replace=True, iterations=brain.repeat)

            assert overlap(assembly_b.representative_neurons, area_b.winners) <= 0.10, \
                "Assemblies formed bi-directional links without merge"

        gc.collect()

    with recipe:
        (assembly_a | assembly_b) >> area_c

    for _ in range(CERTAINTY_REPEAT):
        with bake(recipe, 0.1, Connectome, train_repeat=10, effective_repeat=EFFECTIVE_REPEAT) as brain:
            assembly_a >> area_c
            brain.next_round(subconnectome={area_c: [area_b]}, replace=True, iterations=brain.repeat)

            assert overlap(assembly_b.representative_neurons, area_b.winners) > 0.10, \
                "Assemblies haven't formed bi-directional links"

        gc.collect()


def test_construct_firing_order(area_a, area_b, area_c, stim_a, stim_b, assembly_a, assembly_b, assembly_c):
    correct_firing_order = [{stim_a: [area_a], stim_b: [area_b]}, {assembly_a: [area_c], assembly_b: [area_c]}]
    assert _construct_firing_order([assembly_a, assembly_b], area_c) == correct_firing_order
    assert _construct_firing_order([assembly_c], area_a) == correct_firing_order + [{assembly_c: [area_a]}]
