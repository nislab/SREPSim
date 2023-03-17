"""
Auxiliary functionalities.

Author: Novak Boškov <boskov@bu.edu>
Date: September, 2022.
"""

import os
import subprocess
import itertools
import pickle
import uuid

from typing import List, Set, Tuple, Optional, Generator
from functools import reduce
from operator import mul

import scipy

import networkx as nx
import numpy as np

from tqdm import tqdm

DEFAULT_NAUTY_PATH = os.path.expanduser('~/Downloads/nauty27r4')
NAUTY_TEMP_FILE = '.nauty_tmp'

# custom types
Assign_T = List[Set[int]]


def gen_all_n_vertex_graphs(n: int,
                            nauty_path=DEFAULT_NAUTY_PATH) \
                            -> Generator[nx.Graph, None, None]:
    """
    Generate all non-isomorphic graphs with n vertices.

    Uses Nauty. Calls it in a separate process.

    Parameters:
    --------
    n: int
        The number of vertices.
    nauty_path: str
        Absolute path to the directory where Nauty executables are.
    """
    geng_path = os.path.join(nauty_path, 'geng')
    listg_path = os.path.join(nauty_path, 'listg')

    geng = subprocess.Popen([f"{geng_path}", '-c', f"{n}"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    tmp_f = open(NAUTY_TEMP_FILE, 'w')
    subprocess.call([f"{listg_path}", '-e'], stdin=geng.stdout, stdout=tmp_f)
    tmp_f.close()

    with open(NAUTY_TEMP_FILE, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith('Graph'):
                f.readline()    # drop the next line
                edges = f.readline().strip().split(' '*2)
                G = nx.from_edgelist([(e.split(' ')[0], e.split(' ')[1])
                                      for e in edges])

                # `geng -c` should never generate a disconnected
                # graph. But, it apparently happens. For instance,
                # `geng -c 10` generated ~630 disconnected graphs for
                # me in ~4M graphs (i.e., at ~30% progress).
                if nx.is_connected(G):
                    yield G

            line = f.readline()

    os.remove(NAUTY_TEMP_FILE)


def count_all_n_vertex_graphs(n: int,
                              nauty_path=DEFAULT_NAUTY_PATH) \
                              -> int:
    """
    Count the number of non-isomorphic graphs with `n` vertices.

    Parameters:
    --------
    n: int
        The number of vertices.
    """
    geng_path = os.path.join(nauty_path, 'geng')

    geng = subprocess.Popen([f"{geng_path}", '-c', '-u', f"{n}"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)

    if geng.stderr is not None:
        stderr_s = geng.stderr.read().decode('utf-8')
        pos = stderr_s.find('>Z')
        count = int(stderr_s[pos:].split(' ')[1])
    else:
        raise ValueError("Nauty's 'geng' program did not write to stderr.")

    return count


def n_idx(sizes: List[int]) -> List[List[int]]:
    """
    Generate indices of a ragged matrix.

    The ragged matrix has `len(size)` rows and `i`th row has `sizes[i]`
    columns.

    In returned list, each element is a list `ret` in which `ret[i]`
    ranges between 0 and `sizes[i]` - 1.

    Example:
    >>> n_idx([1, 2, 3])
    [[0, 0, 0],
     [0, 0, 1],
     [0, 0, 2],
     [0, 1, 0],
     [0, 1, 1],
     [0, 1, 2]]

    Parameters:
    --------
    sizes: List[int]
        List of ranges in each dimension.

    Returns:
    --------
        List of `len(sizes)`-long lists.
    """
    if len(sizes) == 1:
        return [[x] for x in range(sizes[0])]

    ret = []
    for i in range(sizes[0]):
        for j in n_idx(sizes[1:]):
            ret.append([i] + j)

    assert len(ret) == reduce(mul, sizes)

    return ret


def is_assign_possible(
        m_2d: np.ndarray = np.array([[-1, 1, 1],
                                     [4, -1, 1],
                                     [1, 1, -1]]),
        max_set_size: int = 5,
        universe_size: int = 10,
        sizes: Optional[List[int]] = None,
        validity: str = 'pair') -> Optional[Assign_T]:
    """
    Brute force to find whether the assignment is possible.

    Parameters:
    ---------
    m_2d: np.ndarray
        2D array of pair-wise differences where `m_2d[i][j]`
        represents `len(S[i].difference(S[j]))`, or the matrix of
        mutual differences (depending on the `validity` parameter).
    max_set_size: int
        Maximum set size to consider.
    universe_size: int
        Set elements lie in 0..universe_size - 1.
    sizes: Optional[List[int]]
        When passed then the assignment also needs to adhere to
        desired set sizes. If `sizes` are given then `max_set_size` is
        set to `max(sizes)` (can be optimized further).
    validity: str
        When 'pair', treat `m_2d` as a pair-wise differences
        matrix, which is default. When 'mut' treat it as a mutual
        differences matrix.

    Return:
    --------
        First assignment that matches, or None if there is no matching
        assignment.
    """
    n, cols = m_2d.shape
    assert n == cols, f"non-square matrix passed, shape: {m_2d.shape}"

    if sizes is not None:
        max_set_size = max(sizes)

    def is_valid(asgn: Assign_T) -> bool:
        if validity == 'pair':
            # check the matrix as if it was a pair-wise differences
            # matrix.
            for i in range(n):
                for j in range(n):
                    if i != j and m_2d[i][j] != -1:
                        d = len(asgn[i].difference(asgn[j]))
                        if d != m_2d[i][j]:
                            return False

            return True
        elif validity == 'mut':
            # check the matrix as if it was a matrix of mutual
            # differences.
            for i in range(n):
                for j in range(n):
                    if i < j and m_2d[i][j] != -1:
                        d = len(asgn[i].difference(asgn[j])) \
                            + len(asgn[j].difference(asgn[i]))
                        if d != m_2d[i][j]:
                            return False

            return True
        else:
            raise ValueError("`validity` must be either 'pair' or 'mut'.")

    def is_matches_sizes(asgn: Assign_T):
        if sizes is None:
            return True

        return [len(x) for x in asgn] == sizes

    set_sizes_combs = itertools.combinations_with_replacement(
        range(max_set_size + 1), n)
    # one cmb is a set of n integers that tells us the sizes of data
    # sets at n nodes.
    for cmb in tqdm(list(set_sizes_combs)):
        # one perm is one set size assignment. For 3 nodes, it could
        # be (2, 3, 1). First node has data set size 2, etc.
        for perm in itertools.permutations(cmb):
            # List of Possible Assignments Per Set
            # len(lpaps) == len(perm) == n.
            # First element of lpaps are all possible set assignments
            # of set 0 that respect perm (e.g., of 2 elements).
            lpaps: List[Assign_T] = []
            for i_size in perm:
                set_cmbs = \
                    [set(x) for x in
                     itertools.combinations(range(universe_size), i_size)]
                lpaps.append(set_cmbs)

            idxs = n_idx([len(x) for x in lpaps])
            for idx in idxs:
                asgn: Assign_T = []
                for i, j in enumerate(idx):
                    asgn.append(lpaps[i][j])

                if is_valid(asgn) and is_matches_sizes(asgn):
                    return asgn


def diff_part_from_asgn(network: nx.Graph,
                        asgn: Assign_T,
                        assume_complete: bool = True) -> np.ndarray:
    """
    Calculate differences partition matrix given an assignment.

    Parameters:
    --------
    network: nx.Graph
        Network as a graph.
    asgn: Assign_T
        Assignment.
    assume_complete: bool
        Same as in `assign_diffs_from_set_sizes`.
    """
    n = len(asgn)

    d_asgn: np.ndarray = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            elif assume_complete:
                d_asgn[i][j] = len(asgn[i].difference(asgn[j]))
            elif i in network[j]:
                d_asgn[i][j] = len(asgn[i].difference(asgn[j]))
            else:
                d_asgn[i][j] = -1

    return d_asgn


def mut_diff_from_asgn(network: nx.Graph,
                       asgn: Assign_T,
                       assume_complete: bool = False,
                       use_np: bool = False) -> np.ndarray:
    """
    Calculate mutual differences matrix given an assignment.

    Parameters:
    --------
    network: nx.Graph
        Network as a graph.
    asgn: Assign_T
        Assignment.
    assume_complete: bool
        Same as in `assign_diffs_from_set_sizes`.
    use_np: bool
        Whether `asgn` is a 2D array where sets are represented as `ndarray`s.
    """
    n = len(asgn)

    d_m: np.ndarray = np.zeros([n, n], dtype=int)
    for i in range(n):
        for j in range(n):
            if i < j:
                if i in network[j] or assume_complete:
                    if use_np:
                        d_m[i][j] = len(np.setdiff1d(asgn[i], asgn[j])) \
                            + len(np.setdiff1d(asgn[j], asgn[i]))
                    else:
                        d_m[i][j] = len(asgn[i].difference(asgn[j])) \
                            + len(asgn[j].difference(asgn[i]))

    return d_m


def assign_diffs_from_set_sizes(dist: scipy.stats.rv_continuous,
                                network: nx.Graph,
                                uni_size: Optional[int] = None,
                                psi: Optional[float] = 2,
                                assume_complete: bool = True,
                                no_matrix: bool = False,
                                use_np: bool = False) \
                                -> Tuple[Assign_T, np.ndarray, np.array]:
    """
    Draw set sizes from a distribution and calculate differences.

    Parameters:
    --------
    dist: scipy.stats.continuous_rv
        Distribution of set sizes.
    network: nx.Graph
        The network.
    uni_size: Optional[int]
        The size of the universe from which to draw set elements.
        When not provided, `psi` is used.
    psi: Optional[float]
        Ratio between `uni_size` and `dist.mean()`.
        Used only when `uni_size` is not passed.
    assume_complete: int
        In calculating the differences matrix, consider all possible pairs
        (like the graph is complete). When this is enabled the differences
        matrix will not contain any -1 values.
    no_matrix: bool
        When enabled, only the assignment is generated but not the matrix.
    use_np: bool
        Use a 2D `np.ndarray` to represent `Assign_T` instead of native
        Python's list of sets. The corresponding `np.ndarray` has
        much smaller memory footprint.

    Return:
    --------
    Tuple of the exact assignment of data sets (as a list of sets),
    a 2D matrix of pair-wise differences count where -1 signalizes
    that nodes are not adjacent, and the exact set sizes sampled from
    the distribution as a vector.
    """
    n = len(network.nodes)

    if not uni_size:
        uni_size = int(psi * dist.mean())

    # pre-sample sizes of sets
    szs = [int(x) for x in dist.rvs(size=n)]
    # pre-sample non-unique elements (repetitions are possible
    # depending on the support of the distribution)
    elms = np.random.uniform(0, uni_size - 1, size=sum(szs))

    # assign concrete sets
    if use_np:
        asgn: np.ndarray = np.empty(n, dtype=object)
    else:
        asgn: Assign_T = []

    i = j = 0
    for asgn_idx, s in enumerate(szs):
        j += s
        set_samples = np.unique(
            np.array([int(x) for x in elms[i:j]], dtype=np.int64))
        i = j

        if use_np:
            asgn[asgn_idx] = set_samples
        else:
            asgn.append(set(set_samples))

    if no_matrix:
        d_asgn = None
    else:
        d_asgn = diff_part_from_asgn(network, asgn, assume_complete)

    return asgn, d_asgn, szs


def gen_2_comp(n: int) -> List[Tuple[int]]:
    """Generate all weak 2-compositions of n."""
    n = int(n)
    compositions: List[Tuple[int]] = []
    for i in range(0, n + 1):
        compositions.append((i, n - i))

    return compositions


def gen_D_candidates(U: np.ndarray) -> List[np.ndarray]:
    """
    Generate D matrix candidates from U.

    Parameters:
    --------
    U: np.ndarray
        Upper triangle of a square matrix whose elements are
        mutual differences.

    Return:
    --------
    List of all possible D matrices generated from the given U.
    """
    def impl(U):
        n, m = U.shape

        # U must be square
        assert n == m, f"{U} is not square."

        # U must be an upper diagonal (without the diagonal itself)
        for i in range(n):
            for j in range(n):
                if i >= j and U[i][j] != 0:
                    raise ValueError(f"Non-zero ({U[i][j]}) at ({i}, {j}).")

        # bottom of recursion is a 1x1 matrix
        if n == 1:
            return [np.array([[0]])]

        # generate all two compositions for the first row of U (starting
        # at 1)
        two_cmps: List[List[Tuple[int, int]]] = []
        two_cmps_sizes: List[int] = []
        for i in range(1, n):
            two_cmp: List[Tuple[int, int]] = gen_2_comp(U[0][i])
            two_cmps.append(two_cmp)
            two_cmps_sizes.append(len(two_cmp))

        matrices: List[np.ndarray] = []
        # create all possible "upper left frames" and combine them with
        # the "inner matrices"
        for indices in n_idx(two_cmps_sizes):
            frame = np.zeros([n, n])
            # create one frame from one collection of 2-compositions
            for i, j in zip(range(0, n - 1), indices):
                fst, snd = two_cmps[i][j]
                frame[0, i + 1] = fst
                frame[i + 1, 0] = snd

            # carve the inner matrix
            inner_U = np.empty([n - 1, n - 1])
            for i in range(1, n):
                for j in range(1, n):
                    inner_U[i - 1][j - 1] = U[i][j]

            # for this frame, combine all the inner matrices
            for inner in impl(inner_U):
                m = frame.copy()
                for i in range(1, n):
                    for j in range(1, n):
                        m[i][j] = inner[i - 1][j - 1]

                matrices.append(m)

        return matrices

    # invoke implementation
    candidates = impl(U)

    # check whether the expected number of candidates is generated
    n, _ = U.shape
    expected = 1
    if n > 1:
        for i in range(n):
            for j in range(n):
                if i < j:
                    expected *= U[i][j] + 1

    assert len(candidates) == expected, \
        f"Generated {len(candidates)} while {expected} is expected."

    return candidates


def check_feasibility_theorem(S: scipy.stats.rv_continuous,
                              P: scipy.stats.rv_continuous,
                              network: nx.Graph,
                              reps: int = 100) -> None:
    """
    Given distributions for S and P, construct (M, s) and check the theorem.

    Parameters:
    --------
    S: scipy.stats.rv_continuous
        Distribution of sizes.
    P: scipy.stats.rv_continuous
        Distribution of mutual differences.
    network: nx.Graph
        Network.
    reps: int
        Number of samplings.
    """
    E = len(network.edges)
    V = len(network.nodes)

    def draw_positive(dist: scipy.stats.rv_continuous, size: int) -> List[int]:
        """Draw positive integers from the distributions."""
        if size == 0:
            return []

        samples = [int(x) for x in dist.rvs(size=size)]
        pos = [x for x in samples if x >= 0]
        return pos + draw_positive(dist, size=len(samples) - len(pos))

    def condition(s: List[int], M: np.ndarray) -> bool:
        for i, j in network.edges:
            if not abs(s[i] - s[j]) <= M[i][j] <= s[i] + s[j]:
                return False

        return True

    def write_counterexample(
            M: np.ndarray,
            s: List[int],
            network: nx.Graph,
            asgn: List[Set[int]] = []) -> None:
        print('There is a counterexample.')
        sfx = uuid.uuid4().hex[:8]
        with open(f"M-s-G-tripple-{sfx}.pickle", 'wb') as f:
            pickle.dump((M, s, network, asgn), f)

    r = 0
    while r < reps:
        # sample an (M, s) pair
        s = draw_positive(S, size=V)
        mut_d = draw_positive(P, size=E)
        M = np.zeros([V, V])
        for i, (u, v) in enumerate(network.edges):
            M[u][v] = mut_d[i]

        checks = condition(s, M)

        if checks:
            continue

        print(f"[{r}]: Theorem: {checks}. "
              "`is_assign_possible` runs with small parameters.")
        asgn = is_assign_possible(M,
                                  sizes=s,
                                  universe_size=8,
                                  validity='mut')

        if not checks and asgn is not None:
            write_counterexample(M, s, network, asgn)

        r += 1
