from brain import *
from non_lazy_brain import *
from lazy_brain import LazyBrain
import numpy as np
# ____// NON LAZY TESTS //____


def bothbrains(func):
    def wrapper():
        func(NonLazyBrain)
        func(LazyBrain)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def test_area_in_brain():
    """test for non lazy brain"""
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert 'a' in brain.areas.keys()
    assert 'a' in brain.connectomes.keys()


def test_stimulus_in_brain():
    """test for non lazy brain"""
    brain = NonLazyBrain(p=0)
    brain.add_stimulus(name='s', k=2)
    assert 's' in brain.stimuli.keys()


def test_init_connectomes_area():
    """test for non lazy brain"""
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert all([all([brain.connectomes['a']['a'][i][j] == 0 for i in range(3)]) for j in range(3)])
    assert brain.areas['a'].area_beta['a'] == 0.1
    brain = NonLazyBrain(p=1)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert all([all([brain.connectomes['a']['a'][i][j] == 1 for i in range(3)]) for j in range(3)])
    assert brain.areas['a'].area_beta['a'] == 0.1


def test_init_connectomes_stimulus():
    """test for non lazy brain"""
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=2)
    assert all([all([brain.stimuli_connectomes['s']['a'][i][j] == 0 for i in range(2)]) for j in range(3)])
    assert brain.areas['a'].stimulus_beta['s'] == 0.1
    brain = NonLazyBrain(p=1)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=2)
    assert all([all([brain.stimuli_connectomes['s']['a'][i][j] == 1 for i in range(2)]) for j in range(3)])
    assert brain.areas['a'].stimulus_beta['s'] == 0.1


@bothbrains
def test_project_support_size(brain_cls):
    for p in [0.3, 0.5]:
        support_size = 0
        brain = brain_cls(p)
        brain.add_area(name='a', n=3, k=1, beta=0.1)
        brain.add_stimulus(name='s', k=2)
        assert brain.areas['a'].support_size == 0
        for _ in range(3):
            brain.project(stim_to_area={'s': ['a']}, area_to_area={})
            support_size += brain.areas['a'].num_first_winners
            assert support_size == brain.areas['a'].support_size


def test_project_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})
    assert brain.areas['a'].winners == [0]


def test_project_num_first_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})
    assert brain.areas['a'].num_first_winners == 1

# Supposed to test whether or not the code crashes with no winners
def test_project_no_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})


def test_project_connectomes():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {})
    brain.project({}, {'a': ['b']})
    assert abs(brain.connectomes['a']['b'][0][0] - 1.1) < 0.0001


# Supposed to test whether or not the code crashes with different n's
def test_project_different_n():
    brain = NonLazyBrain(p=0.5)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.project({'s': ['a']}, {'a': ['b']})


# Supposed to test whether or not the code crashes with different k's
def test_project_different_k():
    brain = NonLazyBrain(p=0.5)
    brain.add_area(name='a', n=30, k=5, beta=0.1)
    brain.add_area(name='b', n=30, k=3, beta=0.1)
    brain.add_stimulus(name='s', k=3)
    brain.project({'s': ['a']}, {'a': ['b']})


def test_project_area_winners():
    """test for lazy brain"""
    brain = LazyBrain(p=0.01)
    brain.add_area(name='a', n=10000, k=100, beta=0.05)
    brain.add_area(name='b', n=10000, k=100, beta=0.05)
    brain.add_stimulus(name='s', k=100)

    brain.project({'s': ['a']}, {})
    brain.project({'s': ['a']}, {'a': ['b']})

    a = brain.areas['a']
    b = brain.areas['b']
    assert len(a.winners) == a.k
    assert len(set(a.winners)) == a.k
    assert len(b.winners) == b.k
    assert len(set(b.winners)) == b.k


def test_project_num_first_winners_lazy():
    """test for lazy brain"""
    brain = LazyBrain(p=0.5)
    brain.add_area(name='a', n=50, k=7, beta=0.1)
    brain.add_area(name='b', n=50, k=7, beta=0.1)
    brain.add_stimulus(name='s', k=7)
    brain.project({'s': ['a']}, {})
    assert brain.areas['a'].num_first_winners == 7

@bothbrains
def test_multiple_stimuli(brain_cls):
    k = 100; n = 10000; beta = 0.05; p = 0.01
    b = brain_cls(p)
    b.add_stimulus('s', k)
    b.add_stimulus('t', k)
    b.add_area('a', n, k, beta)
    b.project({'s': ['a']}, {})
    try:
        b.project({'t': ['a']}, {'a':['a']})
    except:
        print('FAILED test_multiple_stimuli for class' + brain_cls.__name__)

@bothbrains
def test_small_area(brain_cls):
    k = 1; n = 2; beta = 0.05; p = 0.5
    for _ in range(32):
        b = brain_cls(p)
        b.add_stimulus('s', k)
        b.add_area('a', n, k, beta)
        b.add_area('b', n, k, beta)
        try:
            b.project({'s': ['a']}, {})
            b.project({'s': ['b']}, {})
            b.project({'s': ['b']}, {'b': ['a']})
            b.project({'s': ['b']}, {'b': ['b'], 'a': ['a', 'b']})
            b.project({'s': ['a']}, {'b': ['a'], 'a': ['a']})
        except:
            print('FAILED test_small_area for class' + brain_cls.__name__)
            break

