import gc

import pytest

from assembly_calculus.brain import *
from assembly_calculus.brain.connectome import *


EPSILON = 0.001
BASIC_BETA = 0.1
STANDARD_SIZES = [(5000, 71), (1250, 35), (3750, 61), (2500, 50)]
SMALL_SIZES = [(4, 2), (9, 3), (16, 4), (25, 5)]

gc.collect()


def assert_plasticity(conn, source, dest):
    conn = conn.connections[source, dest]
    assert (conn.synapses.max() <= (1 + conn.beta))


def check_synapses(con, value):
    return all([all([con.synapses[i, j] == value for i in range(con.source.n)]) for j in range(con.dest.n)])


def test_area_in_connectome():
    conn = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    assert a in conn.areas


def test_stimulus_in_connectome():
    conn = Connectome(p=0, initialize=True)
    s = Stimulus(n=3, beta=BASIC_BETA)
    conn.add_stimulus(s)
    assert s in conn.stimuli


def test_init_connectomes_area():
    conn = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    assert check_synapses(conn.connections[a, a], 0)
    assert conn.connections[a, a].beta == BASIC_BETA
    conn = Connectome(p=1)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    assert check_synapses(conn.connections[a, a], 1)
    assert conn.connections[a, a].beta == BASIC_BETA


def test_init_connectomes_stimulus():
    conn = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    s = Stimulus(n=2, beta=BASIC_BETA)
    conn.add_stimulus(s)
    assert check_synapses(conn.connections[s, a], 0)
    assert conn.connections[s, a].beta == BASIC_BETA
    conn = Connectome(p=1)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    s = Stimulus(n=2, beta=BASIC_BETA)
    conn.add_stimulus(s)
    assert check_synapses(conn.connections[s, a], 1)
    assert conn.connections[s, a].beta == BASIC_BETA


def simple_conn():
    conn = Connectome(p=0, initialize=True)
    a = Area(n=2, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    b = Area(n=2, k=1, beta=BASIC_BETA)
    conn.add_area(b)
    s = Stimulus(n=1, beta=BASIC_BETA)
    conn.add_stimulus(s)
    conn.connections[a, b].synapses[0, 0] = 1
    conn.connections[a, b].synapses[0, 1] = 0
    conn.connections[a, b].synapses[1, 0] = 0
    conn.connections[a, b].synapses[1, 1] = 0
    conn.connections[s, a].synapses[0, 0] = 1
    conn.connections[s, a].synapses[0, 1] = 0
    return conn, a, b, s


# Supposed to test whether or not the code crashes with no winners
def test_fire_winners():
    conn, a, b, s = simple_conn()
    conn.fire({s: [a], a: [b]})
    assert conn.winners[a] == {0}


def test_fire_connectomes():
    conn, a, b, s = simple_conn()
    conn.fire({s: [a]})
    conn.fire({a: [b]})

    assert abs(conn.connections[a, b].synapses[0, 0] - (1 + BASIC_BETA)) < EPSILON


# Supposed to test whether or not the code crashes with different n's
def test_fire_different_n():
    conn = Connectome(p=0.5, initialize=True)
    a = Area(n=3, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    b = Area(n=2, k=1, beta=BASIC_BETA)
    conn.add_area(b)
    s = Stimulus(n=2, beta=BASIC_BETA)
    conn.add_stimulus(s)
    conn.fire({s: [a], a: [b]})


# Supposed to test whether or not the code crashes with different k's
def test_fire_different_k():
    conn = Connectome(p=0.5, initialize=True)
    a = Area(n=30, k=5, beta=BASIC_BETA)
    b = Area(n=30, k=3, beta=BASIC_BETA)
    conn.add_area(a)
    conn.add_area(b)
    s = Stimulus(n=3, beta=BASIC_BETA)
    conn.add_stimulus(s)
    conn.fire({s: [a], a: [b]})


@pytest.mark.parametrize("n, k", STANDARD_SIZES)
def test_area_winner_count(n, k):
    conn = Connectome(p=1, initialize=True)
    a = Area(n=n, k=k, beta=BASIC_BETA)
    conn.add_area(a)
    conn.fire({a: [a]})
    assert len(conn.winners[a]) == k


@pytest.mark.parametrize("n, k", STANDARD_SIZES)
def test_areas_winner_count(n, k):
    conn = Connectome(p=1, initialize=True)
    a = Area(n=n, k=k+1, beta=BASIC_BETA)
    conn.add_area(a)
    b = Area(n=n, k=k, beta=BASIC_BETA)
    conn.add_area(a)
    conn.add_area(b)
    conn.fire({a: [b]})
    assert len(conn.winners[b]) == k


@pytest.mark.parametrize("count", [2, 3, 5, 10, 25])
@pytest.mark.parametrize("n, k", SMALL_SIZES)
def test_fire_many_to_one(n, k, count):
    conn = Connectome(p=1, initialize=True)
    a = Area(n=n, k=k, beta=BASIC_BETA)
    conn.add_area(a)
    bs = [Area(n=n, k=k, beta=BASIC_BETA) for _ in range(count)]
    for b in bs:
        conn.add_area(b)
    conn.fire({b: [a] for b in bs})
    assert len(conn.winners[a]) == k
    for b in bs:
        assert_plasticity(conn, b, a)


@pytest.mark.parametrize("count", [2, 3, 5, 10, 25])
@pytest.mark.parametrize("n, k", SMALL_SIZES)
def test_fire_one_to_many(n, k, count):
    conn = Connectome(p=1, initialize=True)
    a = Area(n=n, k=k, beta=BASIC_BETA)
    conn.add_area(a)
    bs = [Area(n=n, k=k, beta=BASIC_BETA) for _ in range(count)]
    for b in bs:
        conn.add_area(b)
    conn.fire({a: bs})
    for b in bs:
        assert len(conn.winners[b]) == k
        assert_plasticity(conn, a, b)


@pytest.mark.xfail()
def test_bigger_k():
    conn = Connectome(p=1, initialize=True)

    a = Area(n=1, k=5, beta=BASIC_BETA)
    conn.add_area(a)
    conn.fire({a: [a]})


@pytest.mark.xfail()
def test_n_is_negative():
    conn = Connectome(p=1, initialize=True)

    a = Area(n=-10, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    conn.fire({a: [a]})


@pytest.mark.xfail()
def test_p_overflow():
    conn = Connectome(p=5, initialize=True)

    a = Area(n=-10, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    conn.fire({a: [a]})


@pytest.mark.xfail()
def test_p_underflow():
    conn = Connectome(p=-5, initialize=True)

    a = Area(n=-10, k=1, beta=BASIC_BETA)
    conn.add_area(a)
    conn.fire({a: [a]})

