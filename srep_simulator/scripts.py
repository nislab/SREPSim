"""
Various long-running scripts.

Author: Novak Boškov <boskov@bu.edu>
Date: September, 2022.
"""


import math
import pickle
import uuid
import time
import sys
import gc

import scipy

import networkx as nx
import numpy as np
import pandas as pd

import srep_simulator.tvg as tvg

from scipy.stats import norm

from typing import List, Tuple, Dict, Set, Any
from tqdm import tqdm

from srep_simulator.srep import SREPSimulator, SREPSimulator_tvg, me_srep_analytic
from srep_simulator.aux import gen_all_n_vertex_graphs, \
    count_all_n_vertex_graphs, assign_diffs_from_set_sizes, mut_diff_from_asgn


def theorem_2_for_big_graphs(graph_sizes: List[int] = [9, 10]) -> None:
    """Run theorem 2 validation for huge graphs."""
    def pickle_network(sim: SREPSimulator, f_name: str) -> None:
        with open(f_name, 'wb') as f:
            pickle.dump(sim.network, f)

    comm_costs = []  # type: List[Tuple[float, float]]
    for gs in tqdm(graph_sizes):
        # type: Generator[nx.Graph, None, None]
        graphs = gen_all_n_vertex_graphs(n=gs)
        total_graphs = count_all_n_vertex_graphs(n=gs)
        cc_min = math.inf
        cc_max = -math.inf
        for G in tqdm(graphs, total=total_graphs):
            s = SREPSimulator(network=G, trace_file='/dev/null')
            s.run()
            cc = s.get_stats().communication_cost

            if cc < cc_min:
                cc_min = cc
                pickle_network(s, f"{gs}_nodes_min_comm.pickle")
            if cc > cc_max:
                cc_max = cc
                pickle_network(s, f"{gs}_nodes_max_comm.pickle")

        comm_costs.append((cc_min, cc_max))

    out_file = (f"{theorem_2_for_big_graphs.__name__}"
                f"_{'_'.join([str(x) for x in graph_sizes])}"
                f"_{uuid.uuid4().hex[:8]}.pickle")
    with open(out_file, 'wb') as f:
        pickle.dump(comm_costs, f)


ws_kp_pair = Tuple[int, int]
comm_and_red = Tuple[int, int]


def communication_cost_ws(
        graph_sizes: List[int] = [4, 5, 6, 7, 8, 9, 10],
        reps: int = 1000,
        ws_kps: List[ws_kp_pair] = [(3, 0.6)]) -> None:
    """Find communication costs for WS graphs."""
    begin_t = time.time()

    stats: Dict[Tuple[int, ws_kp_pair], List[int]] = {}
    for gs in tqdm(graph_sizes):
        for ws_kp in tqdm(ws_kps):
            ls: List[comm_and_red] = []
            for _ in tqdm(range(reps)):
                s = SREPSimulator(ws_nkp=(gs, *ws_kp), trace_file='/dev/null')
                s.run(timeout=0)
                cc = s.get_stats()
                ls.append((cc.communication_cost, cc.redundant_trans))

            stats[(gs, ws_kp)] = ls

    out_file = (f"{communication_cost_ws.__name__}"
                f"_{'_'.join([str(x) for x in graph_sizes])}"
                f"_{uuid.uuid4().hex[:8]}.pickle")
    with open(out_file, 'wb') as f:
        pickle.dump(stats, f)

    print(f"Elapsed: {(time.time() - begin_t) / 60:.2f} min.",
          file=sys.stderr)


def debug():
    """
    Debug the theorem.

    The old version of the upper bound in elementary parallel SREP
    communication.
    """
    for gs in range(10, 110):
        print(f"---> for {gs}")
        s = SREPSimulator(ws_nkp=(gs, 8, 0.6), trace_file='/dev/null')
        s.run()
        stat = s.get_stats()
        if stat.communication_cost > gs * ((2 * gs) - 1):
            break

    return s


def theorem_3(graph_sizes=[10], find_first=True):
    """Help prove the T_100 theorem for elementary parallel SREP."""
    for gs in tqdm(graph_sizes):
        count = count_all_n_vertex_graphs(gs)
        for G in tqdm(gen_all_n_vertex_graphs(gs), total=count):
            s = SREPSimulator(network=G, trace_file='/dev/null')
            s.run()
            end_gen = s.get_stats().end_gen
            diameter = nx.diameter(G)
            graph_size = len(G.nodes)
            if end_gen + 1 != diameter:
                print(f"---> end_gen: {end_gen},"
                      f" diameter: {diameter},"
                      f" graph_size: {graph_size}")

                suffix = uuid.uuid4().hex[:8]
                with open(f"diam_not_end_gen_{suffix}.pickle", 'wb') as f:
                    pickle.dump(G, f)

                if find_first:
                    return


def sigma_R_E_theorem(graph_size=10, reps=10, find_n=10):
    """Help prove the theorem with Sigma, R, E relation."""
    found = 0

    count = count_all_n_vertex_graphs(graph_size)
    graphs = gen_all_n_vertex_graphs(graph_size)
    for gs in tqdm(graphs, total=count):
        for _ in range(reps):
            s = SREPSimulator(network=gs, trace_file='/dev/null')
            s.run()
            stats = s.get_stats()
            rounds = stats.end_gen + 1
            edges = len(s.network.edges)
            if rounds * edges < stats.sync_invocations \
               or rounds != nx.diameter(s.network):
                print(f"rounds: {rounds}, edges: {edges}, "
                      f"invocations: {stats.sync_invocations}, "
                      f"diff: {(rounds * edges) - stats.sync_invocations}\n"
                      f"diameter: {nx.diameter(s.network)}")

                suffix = uuid.uuid4().hex[:8]
                with open(f"Sigma_R_E_exception_{suffix}.pickle", 'wb') as f:
                    pickle.dump(s.network, f)

                found += 1
                if found == find_n:
                    return


def gen_elems(
        G: nx.Graph,
        dist: scipy.stats.rv_continuous = scipy.stats.genhyperbolic(
            **{'p': 0.8290151067086478,
               'a': 3.2438523525534995e-08,
               'b': 2.1137430267794334e-08,
               'loc': 1380.9999929285977,
               'scale': 9.473594438724517e-06})) -> List[Set[int]]:
    """
    Assign data to graph nodes given a distribution of differences.

    Given an empirical distribution of mutual differences among two nodes,
    compute the data sets that each node in the network holds.

    Parameters:
    --------
    G: nx.Graph
        The network.
    dist: scipy.stats.rv_continuous
        Distribution of differences between two nodes.
    """
    n = len(G.nodes)
    e = len(G.edges)

    # Generate the matrix of diff counts
    d_m = np.zeros([n, n])
    d_sample = dist.rvs(size=2*e)
    for i, (u, v) in enumerate(G.edges):
        d_m[u][v] = int(d_sample[2 * i])
        d_m[v][u] = int(d_sample[2 * i + 1])

    return d_m


def mc_me_srep(ws_params: List[Tuple[int, int, float]],
               reps: int,
               S: scipy.stats.rv_continuous,
               psi: float) -> None:
    """
    Monte Carlo ME-SREP.

    Parameters:
    --------
    ws_params: List[Tuple[int, int, float]]
        Watts-Strogatz network parameters.
    reps: int
        Number of repetitions to run for each experiment.
    S: scipy.stats.rv_continuous
        Distribution of set sizes to use in `assign_diffs_from_set_sizes`.
    psi: float
        Psi to use in `assign_diffs_from_set_sizes`.
    """
    df = pd.DataFrame(
        columns=['ws', 'cc', 'rt', 'et', 'eg', 'si', 'r_time', 'diam'])

    for ws_p in ws_params:
        print(f"Watts-Strogatz: {ws_p}")
        for _ in tqdm(range(reps)):
            s = SREPSimulator(ws_nkp=ws_p,
                              me_srep_dist_psi=(S, psi),
                              trace_file='/dev/null')
            s.run(timeout=0)

            st = s.get_stats()
            diam = nx.diameter(s.network)
            df = pd.concat([df, pd.DataFrame({'ws': [ws_p],
                                              'cc': [st.communication_cost],
                                              'rt': [st.redundant_trans],
                                              'et': [st.end_timer],
                                              'eg': [st.end_gen],
                                              'si': [st.sync_invocations],
                                              'r_time': [st.real_time],
                                              'diam': [diam]})],
                           ignore_index=True)

            del s
            gc.collect()

    # convert column types
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c])

    suffix = uuid.uuid4().hex[:8]
    with open(f"mc_me_srep_{suffix}.pickle", 'wb') as f:
        pickle.dump(df, f)


def theorem_5(degs: List[int] = [2, 4, 6],
              network_sizes: List[int] = [1000],
              psis: List[float] = [0.355, 0.6],
              dist: scipy.stats.rv_continuous = scipy.stats.maxwell(
                  **{'loc': 15401.20304028427, 'scale': 15920.396446893377}),
              iters: int = 10):
    """
    Compare results from event-based simulator and Theorem 5.

    Parameters:
    --------
    iters: int
        Iterations for each experiment.
    """
    pickle_sfx = uuid.uuid4().hex[:8]

    for n in network_sizes:
        for progress, deg in enumerate(degs):
            print(f"Progress: {progress + 1}/{len(degs)} degree.")
            for psi in psis:
                for it in range(iters):
                    G = nx.connected_watts_strogatz_graph(n, deg, 0.24)
                    asgn, _, _ = assign_diffs_from_set_sizes(
                        dist, G, psi=psi, no_matrix=True, use_np=True)

                    asgn_l = [set(x) for x in asgn]

                    sim = SREPSimulator(network=G,
                                        me_srep_dist_psi=asgn_l,
                                        trace_file='/dev/null')
                    sim.run(timeout=0)
                    sim_stats = sim.get_stats()
                    del sim
                    del asgn_l
                    gc.collect()

                    th = me_srep_analytic(network=G, asgn=asgn)

                    record = {'sim_stats': sim_stats,
                              'theorem_stats': th,
                              'diameter': nx.diameter(G),
                              'asgn': asgn,
                              'network_size': n,
                              'avg_deg': deg,
                              'psi': psi,
                              'iter_count': it}

                    f_n = f"theorem_5__{n}_{deg}_{psi}_{it}_{pickle_sfx}.pickle"
                    with open(f_n, "wb") as f:
                        pickle.dump(record, f)

                    del asgn
                    gc.collect()


def srep_vs_mempool(
        net_sizes=[50, 100, 200, 300, 400, 500, 600],
        psis=[0.3, 0.355, 0.4, 0.45, 0.5, 0.55, 0.6],
        mempool_sync_params=[(0.05, 1000, 10), (0.1, 1000, 10), (0.2, 1000, 10), (0.3, 1000, 10)],
        reps=10,
        ws_deg_p: Tuple[int, float]=(4, 0.24)) -> None:
    """Comparison between SREP and MempoolSync."""

    S = scipy.stats.maxwell(**{'loc': 15401.20304028427, 'scale': 15920.396446893377})

    res: List[Dict[str: Any]] = []

    for ns in tqdm(sorted(net_sizes, reverse=True)):
        for psi in psis:
            for y, def_tx, much_larger in mempool_sync_params:
                for _ in range(reps):
                    sim = SREPSimulator(ws_nkp=(ns, *ws_deg_p),
                                        me_srep_dist_psi=(S, psi),
                                        mempoolsync_params=(y, def_tx, much_larger),
                                        trace_file='/dev/null')

                    sim.run(timeout=0)
                    stats = sim.get_stats()
                    del sim
                    gc.collect()

                    res.append({'stats': stats,
                                'mempool': (y, def_tx, much_larger),
                                'psi': psi,
                                'network_size': ns,
                                'ws_deg_p': ws_deg_p})

    sfx = uuid.uuid4().hex[:8]
    f_n = f"srep_vs_mempool_{sfx}.pickle"
    with open(f_n, "wb") as f:
        pickle.dump(res, f)


def sim_experiments(
        net_sizes: List[int] = [1000],
        avg_degs: List[int] = [8, 6, 4, 2],
        reps: int = 10,
        S: scipy.stats.rv_continuous = scipy.stats.maxwell(**{'loc': 15401.20304028427,
                                                              'scale': 15920.396446893377}),
        psi: float = 0.355):
    records: List[Dict[str, Any]] = []

    for ns in net_sizes:
        for deg in tqdm(avg_degs):
            for rep in range(reps):
                sim = SREPSimulator(ws_nkp=(ns, deg, 0.24),
                                    me_srep_dist_psi=(S, psi),
                                    trace_file='/dev/null')
                sim.run(timeout=0)
                stats = sim.get_stats()

                M = mut_diff_from_asgn(sim.network, sim._s_vec)

                record = {'stats': stats,
                          'diam': nx.diameter(sim.network),
                          'M_sum': M.sum(),
                          'M_max': M.max(),
                          'network_size': ns,
                          'avg_deg': deg,
                          'rep': rep}

                records.append(record)

                del sim
                gc.collect()

    sfx = uuid.uuid4().hex[:8]
    f_n = f"sim_experiments_{sfx}.pickle"
    with open(f_n, "wb") as f:
        pickle.dump(records, f)


def analytical_large_net(
        net_size: int = 10_000,
        degs: List[int] = [4, 8, 12, 16, 20, 24, 28],
        psis: List[float] = [0.4, 0.355, 0.3],
        ws_p: float = 0.24,
        reps = 5,
        S: scipy.stats.rv_continuous = scipy.stats.maxwell(**{'loc': 15401.20304028427,
                                                              'scale': 15920.396446893377})):
    records: List[Dict[str, Any]] = []

    for deg in tqdm(degs, desc='Degrees'):
        for psi in tqdm(psis, desc='Psis'):
            for rep in tqdm(range(reps), desc='Repetitions'):
                G = nx.connected_watts_strogatz_graph(net_size, deg, ws_p)
                asgn, _, _ = assign_diffs_from_set_sizes(dist=S,
                                                         network=G,
                                                         psi=psi,
                                                         no_matrix=True,
                                                         use_np=True)

                M = mut_diff_from_asgn(G, asgn, use_np=True)

                stats = me_srep_analytic(network=G, asgn=asgn)

                record = {'stats': stats,
                          'diam': nx.diameter(G),
                          'deg': deg,
                          'network_size': net_size,
                          'psi': psi,
                          'rep': rep,
                          'M_sum': M.sum(),
                          'M_max': M.max()}

                records.append(record)


    sfx = uuid.uuid4().hex[:8]
    f_n = f"analytical_large_net_{sfx}.pickle"
    with open(f_n, "wb") as f:
        pickle.dump(records, f)

def sim_tvg(
        net_sizes: List[int] = list(range(2, 20)),
        avg_degs: List[int] = [1],
        reps: int = 1000):

    for ns in net_sizes:
        records: List[Dict[str, Any]] = []
        timer_list = []
        for deg in tqdm(avg_degs):
            for rep in range(reps):
                sim = SREPSimulator_tvg(ws_nkp=(ns, deg, 0.24),
                                    trace_file='/dev/null')
                sim.run(timeout=0)
                stats = sim.get_stats()

                record = {'stats': stats,
                          'network_size': ns,
                          'rep': rep}
                records.append(record)
                timer_list.append(record['stats'].end_timer)

                del sim
                gc.collect()
        mean = np.mean(timer_list)
        stddev = np.std(timer_list)

        z = norm.ppf(0.975)
        interval = [mean - z * (stddev / np.sqrt(reps)), mean + z * (stddev / np.sqrt(reps))]
        print("Average time:", mean)
        print("Confidence Interval:", interval)
        with open("data.txt", "a") as file:
            file.write("{} {} {}\n".format(ns, mean, interval))

    sfx = uuid.uuid4().hex[:8]
    f_n = f"analytical_large_net_{sfx}.pickle"
    with open(f_n, "wb") as f:
        pickle.dump(records, f)

def sim_tvl(net_sizes: List[int] = list(range(2, 20)),
            reps: int = 1000):
    for ns in net_sizes:
        records: List[Dict[str, Any]] = []
        timer_list = []
        for rep in range(reps):
            graph, time_arr = tvg.generate_tvl(ns)
            sim = SREPSimulator_tvg(network=graph, time_stamp=time_arr,
                                trace_file='/dev/null')
            sim.run(timeout=0)
            stats = sim.get_stats()

            record = {'stats': stats,
                        'network_size': ns,
                        'rep': rep}
            records.append(record)
            timer_list.append(record['stats'].end_timer)

            del sim
            gc.collect()
        mean = np.mean(timer_list)
        stddev = np.std(timer_list)

        z = norm.ppf(0.975)
        interval = [mean - z * (stddev / np.sqrt(reps)), mean + z * (stddev / np.sqrt(reps))]
        print("Average time:", mean)
        print("Confidence Interval:", interval)
        with open("data.txt", "a") as file:
            file.write("{} {} {}\n".format(ns, mean, interval))

    sfx = uuid.uuid4().hex[:8]
    f_n = f"analytical_large_net_{sfx}.pickle"
    with open(f_n, "wb") as f:
        pickle.dump(records, f)

def overnight():
    sim_tvl()

overnight()