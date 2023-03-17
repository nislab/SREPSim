"""
Minimal functional testing of the simulator.

Author: Novak Boškov <boskov@bu.edu>
Date: September, 2022.
"""
import networkx as nx
from srep_simulator.srep import SREPSimulator
from test_aux import stat_compare


def test_triangle():
    G = nx.from_edgelist([(0, 1), (0, 2), (1, 2)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(), dict(communication_cost=6,
                                            end_timer=2,
                                            end_gen=0))


def test_3_node_line():
    G = nx.from_edgelist([(0, 1), (1, 2)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(),
                        dict(communication_cost=6,
                             end_timer=3,
                             end_gen=1,
                             sync_invocations=2 * len(G.edges)))


def test_5_node_A():
    """The spear graph."""
    G = nx.from_edgelist([(0, 2), (1, 2), (1, 4), (2, 4), (3, 4)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(),
                        dict(communication_cost=20,
                             end_timer=5,
                             end_gen=2,
                             sync_invocations=3 * len(G.edges)))


def test_5_node_B():
    """The house graph."""
    G = nx.from_edgelist([(0, 2), (1, 2), (1, 4), (2, 4), (3, 4), (0, 3)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(),
                        dict(communication_cost=24,
                             end_timer=5,
                             end_gen=1,
                             sync_invocations=2 * len(G.edges)))


def test_5_node_C():
    """The three triangles graph."""
    G = nx.from_edgelist([(0, 2), (1, 2), (1, 4),
                          (2, 4), (3, 4), (0, 3), (2, 3)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(),
                        dict(communication_cost=24,
                             end_timer=4,
                             end_gen=1,
                             sync_invocations=2 * len(G.edges)))


def test_5_node_D():
    """The complete graph."""
    G = nx.from_edgelist([(0, 2), (1, 2), (1, 4), (2, 4), (3, 4),
                          (0, 3), (2, 3), (0, 4), (0, 1), (1, 3)])
    s = SREPSimulator(network=G)
    s.run()

    assert stat_compare(s.get_stats(),
                        dict(communication_cost=20,
                             end_timer=2,
                             end_gen=0,
                             sync_invocations=1 * len(G.edges)))
