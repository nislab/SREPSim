"""
SREP Simulator.

Author: Novak Boškov <boskov@bu.edu>
Date: September, 2022.
"""

import math
import signal
import uuid
import pickle
import time
import logging
import itertools

import collections.abc as cabc

from dataclasses import dataclass, field
from queue import PriorityQueue
from abc import ABC, abstractmethod
from typing import List, Tuple, Set, Dict, Any, Optional, Union, Iterable

import scipy
import tqdm

import numpy as np
import networkx as nx

import srep_simulator.aux as aux
import srep_simulator.tvg as tvg

from srep_simulator.aux import Assign_T

MIN_GENERATION_CONST = 0

# types
DistUniSize_T = Tuple[scipy.stats.rv_continuous, float]

# logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: [%(asctime)s]: %(message)s')


@dataclass
class NetworkNode:
    """
    Representation of a node in the network.

    Attributes:
    --------
    idx: int
        Index of the node in the network.
    data_set: Set[int]
        Data kept at the node.
    replicas: Dict[int, Set[int]]
        Synchronization replicas. Neighbor to replica mapping.
    synced_with_me: Dict[int, int]
        Map between nodes that synced with me to the generation
        in which that happened last time.
    _max_coll_len: int = 5
        Truncate point for representing collections in attributes.
    """

    idx: int
    data_set: Set[int] = field(default_factory=set)
    replicas: Dict[int, Set[int]] = field(default_factory=dict)
    synced_with_me: Dict[int, int] = field(default_factory=dict)
    _max_coll_len: int = 5

    def __repr__(self):
        """Representation."""
        def trnc(c: Iterable[Any]) -> str:
            """Truncate long nested collections."""
            if isinstance(c, set):
                lft, rgt = '{', '}'
            elif isinstance(c, list):
                lft, rgt = '[', ']'
            else:
                raise NotImplementedError(f"Can't trnc a {type(c)}.")

            fst = [trnc(x) if isinstance(x, cabc.Iterable) else repr(x)
                   for x in itertools.islice(c, self._max_coll_len)]
            return (f"{lft}{', '.join(fst)}"
                    f"{' ...' if len(c) > self._max_coll_len else ''}{rgt}")

        replicas = {nbr: trnc(s) for nbr, s in self.replicas.items()}

        return (f"NetworkNode(idx={self.idx}, "
                f"data_set={trnc(self.data_set)}, "
                f"replicas={replicas}, "
                f"synced_with_me={self.synced_with_me})")


@dataclass
class Event(ABC):
    """
    Generic event.

    Attributes:
    --------
    current_time: int
        When this event starts.
    simulator:
        The reference to the simulator object.
    completion_time: Optional[int]
        When this event ends.
    duration: Optional[int]
        How long the execution of this event takes.
    """

    current_time: int
    simulator: Any = field(repr=False)  # Reference to the SREPSimulator
    completion_time: Optional[int] = None
    duration: Optional[int] = None

    def __post_init__(self):
        """Set the calculated fields."""
        self.completion_time = self.current_time + (self.duration or 0)

    @abstractmethod
    def apply(self):
        """
        Apply the state changes resulting from execution of this event.

        Returns new events.
        """


@dataclass
class NodeSyncEvent(Event):
    """
    Synchronization loop at the node.

    Attributes:
    --------
    node: int
        The node to sync itself with its neighbors.
    generation: int
        How many times this node synchronized before.
    set_duration: bool
        Whether to calculate differences according to current state.
    """

    node: int = 0
    generation: int = 0
    set_duration: bool = False

    def __post_init__(self):
        """Calculate the duration if needed."""
        if self.set_duration:
            self.duration = self.simulator.calc_duration(self.node)

        super().__post_init__()

    def adjust_duration(self):
        """
        Set the duration of this event according to the current state.

        This is used for the objects that did not set their duration
        at the construction time.
        """
        assert not self.set_duration, \
            f"Duration of {self} has been set at construction time."

        self.duration = self.simulator.calc_duration(self.node)
        self.completion_time += self.duration

    def apply(self) -> List[Event]:
        """Simulate one node synchronization loop."""
        G = self.simulator.network  # type: nx.Graph

        # sync with all neighbors using replicas
        for n in G.neighbors(self.node):
            this_node = G.nodes[self.node]['data']  # type: NetworkNode
            neighbor = G.nodes[n]['data']           # type: NetworkNode

            # sync only the neighbors that did not sync with me in
            # this same generation
            if n in this_node.synced_with_me \
               and this_node.synced_with_me[n] == self.generation:
                continue

            # calculate what this sync will cost in communication if
            # it was done in the MempoolSync-style
            if self.simulator.mempoolsync_params is not None:
                self.simulator._stats.mempoolsync_cc += \
                    self.simulator.calc_mempoolsync_cc(self.node)

            # update the overall communication cost of the simulation
            this_minus_neighbor = this_node.replicas[n].difference(
                neighbor.replicas[self.node])
            neighbor_minus_this = neighbor.replicas[self.node].difference(
                this_node.replicas[n])
            diffs = len(this_minus_neighbor) + len(neighbor_minus_this)
            self.simulator._stats.communication_cost += diffs

            # actually synchronize replicas
            new = this_node.replicas[n].union(neighbor.replicas[self.node])
            this_node.replicas[n] = new
            neighbor.replicas[self.node] = new.copy()

            # note that we have synced with this neighbor
            neighbor.synced_with_me[self.node] = self.generation

            # increment global sync invocation counter
            self.simulator._stats.sync_invocations += 1

        # count the redundant transfers
        d_set = G.nodes[self.node]['data'].data_set
        replicas = G.nodes[self.node]['data'].replicas
        news: List[int] = []
        for _, r in replicas.items():
            news += list(r.difference(d_set))

        self.simulator._stats.redundant_trans += len(news) - len(set(news))

        # unionize all replicas into the data set
        G.nodes[self.node]['data'].data_set = \
            G.nodes[self.node]['data'].data_set.union(
                *G.nodes[self.node]['data'].replicas.values())

        # recreate the replicas
        for i in G.nodes[self.node]['data'].replicas.keys():
            G.nodes[self.node]['data'].replicas[i] = \
                G.nodes[self.node]['data'].data_set.copy()

        # create the next loop's event
        next_loop = NodeSyncEvent(
            node=self.node,
            current_time=self.completion_time,
            generation=self.generation + 1,
            simulator=self.simulator)
        return [next_loop]
    
@dataclass
class NodeSyncEvent_tvg(Event):
    """
    Synchronization loop at the node.

    Attributes:
    --------
    node: int
        The node to sync itself with its neighbors.
    generation: int
        How many times this node synchronized before.
    set_duration: bool
        Whether to calculate differences according to current state.
    time_stamp: dict
        Time stamps when connection status od edges change
    """

    node: int = 0
    generation: int = 0
    set_duration: bool = False
    time_stamp: List[np.ndarray] = None

    def __post_init__(self):
        """Calculate the duration if needed."""
        if self.set_duration:
            self.duration = self.simulator.calc_duration(self.node)

        super().__post_init__()

    def adjust_duration(self):
        """
        Set the duration of this event according to the current state.

        This is used for the objects that did not set their duration
        at the construction time.
        """
        assert not self.set_duration, \
            f"Duration of {self} has been set at construction time."

        self.duration = self.simulator.calc_duration(self.node)
        # print("Duration:", self.duration)
        self.completion_time += self.duration

    def apply(self) -> List[Event]:
        """Simulate one node synchronization loop."""
        # print("==============NEW EVENT=================")

        # update the network according to time_stamp
        self.simulator.network = tvg.update_graph(self.simulator.network, self.time_stamp, self.current_time)

        G = self.simulator.network  # type: nx.Graph
        # print("Self node:", self.node)
        # print("Neighbors:", list(G.neighbors(self.node)))

        self.duration = self.simulator.calc_duration(self.node)
        # print("Duration:", self.duration)
        self.completion_time += self.duration

        # sync with all neighbors using replicas
        for n in G.neighbors(self.node):
            this_node = G.nodes[self.node]['data']  # type: NetworkNode
            neighbor = G.nodes[n]['data']           # type: NetworkNode
            # print("self:", this_node)
            # print("neighbor", neighbor)

            # sync only the neighbors that did not sync with me in
            # this same generation
            if n in this_node.synced_with_me \
               and this_node.synced_with_me[n] == self.generation:
                continue

            # calculate what this sync will cost in communication if
            # it was done in the MempoolSync-style
            if self.simulator.mempoolsync_params is not None:
                self.simulator._stats.mempoolsync_cc += \
                    self.simulator.calc_mempoolsync_cc(self.node)

            # update the overall communication cost of the simulation
            if n in this_node.replicas and self.node in neighbor.replicas:
                this_minus_neighbor = this_node.replicas[n].difference(
                    neighbor.replicas[self.node])
                neighbor_minus_this = neighbor.replicas[self.node].difference(
                    this_node.replicas[n])
                diffs = len(this_minus_neighbor) + len(neighbor_minus_this)
                self.simulator._stats.communication_cost += diffs
            # elif n in this_node.replicas:
            #     diffs = len(this_node.replicas[n])
            #     self.simulator._stats.communication_cost += diffs
            # elif self.node in neighbor.replicas:
            #     diffs = len(neighbor.replicas[self.node])
            #     self.simulator._stats.communication_cost += diffs

            # actually synchronize replicas
            if n in this_node.replicas and self.node in neighbor.replicas:
                new = this_node.replicas[n].union(neighbor.replicas[self.node])
                this_node.replicas[n] = new
                neighbor.replicas[self.node] = new.copy()

            # note that we have synced with this neighbor
            neighbor.synced_with_me[self.node] = self.generation

            # increment global sync invocation counter
            self.simulator._stats.sync_invocations += 1
            # print("Sync cnt:", self.simulator._stats.sync_invocations)

        # count the redundant transfers
        d_set = G.nodes[self.node]['data'].data_set
        replicas = G.nodes[self.node]['data'].replicas
        news: List[int] = []
        for _, r in replicas.items():
            news += list(r.difference(d_set))

        self.simulator._stats.redundant_trans += len(news) - len(set(news))

        # unionize all replicas into the data set
        G.nodes[self.node]['data'].data_set = \
            G.nodes[self.node]['data'].data_set.union(
                *G.nodes[self.node]['data'].replicas.values())

        # recreate the replicas
        for i in G.nodes[self.node]['data'].replicas.keys():
            G.nodes[self.node]['data'].replicas[i] = \
                G.nodes[self.node]['data'].data_set.copy()
            
        # print("Event done:")
        for n in self.simulator.network.nodes:
            this = self.simulator.network.nodes[n]['data'].data_set
            # print(len(this), end=' ')

        # create the next loop's event
        next_loop = NodeSyncEvent_tvg(
            node=self.node,
            current_time=self.completion_time + 1,
            generation=self.generation + 1,
            simulator=self.simulator,
            time_stamp=self.time_stamp)
        return [next_loop]


class EventQ(PriorityQueue):
    """Infinite priority queue."""

    def __init__(self):
        """Create `EventQ`."""
        super().__init__()

    @dataclass(order=True)
    class PrioritizedItem():
        """Pair of integer priority and `Event`."""

        priority: int
        item: Any = field(compare=False)

    def enqueue(self, events: List[Event]):
        """Put the list of events into the queue."""
        for event in events:
            assert event.completion_time is not None, \
                f"Event {event} has no completion time."

            self.put(self.PrioritizedItem(priority=event.completion_time,
                                          item=event))

    def dequeue(self):
        """Get the item with the highest priority."""
        return self.get().item

    def purge(self):
        """Purge the queue."""
        while not self.empty():
            self.get()


@dataclass
class Stats:
    """
    Object to collect statistics from the simulation.

    Parameters:
    --------
    communication_cost: int
        Overall communication cost.
    redundant_trans: int
        When a node receives an element via multiple replicas
        in the same round, all but one transmission is redundant.
    end_timer: int
        Timer value at the end of simulation.
    end_gen: int
        The maximum generation of events that have been executed
        before the simulation got interrupted.
    sync_invocations:
        Overall number of (primitive) sync invocations executed so far.
    real_time: float
        Duration of the simulation in real time seconds.
    mempoolsync_cc: int
        Overall communication that MempoolSync would incur with
        the same amount of `NodeSyncEvent`s.
    """

    communication_cost: int = 0
    redundant_trans: int = 0
    end_timer: float = -math.inf
    end_gen: float = -math.inf
    sync_invocations: int = 0
    real_time: float = 0
    mempoolsync_cc:int = 0


@dataclass
class SREPSimulator():
    """
    The main simulator class.

    Attributes:
    ---------
    network: nx.Graph
        Topology.
    ws_nkp: Tuple[float, float, float]
        Parameters to
        `nx.generators.random_graphs.connected_watts_strogatz_graph`.
    eventq: EventQ
        Simulation event queue.
    timer: int
        Simulation timer.
    max_time: int
        Simulated time after which the simulation will get interrupted.
    trace_file: Optional[str]
        Path to log file that contains traces. If `None` print to stdout.
    me_srep_dist_psi: Optional[Union[DistUniSize_T, Assign_T]]
        When passed, multi elements SREP is initialized instead
        of the default single element SREP. `DistUniSize_T` is a tuple of
        `scipy.stats.rv_continuous` and `float` that represents the psi parameter.
        When an `Assign_T` (list of sets) is passed instead, it is used
        directly to initialize the data sets.
    me_srep_assume_complete: Optional[bool]
        When enabled, pass `assume_complete=True` in `aux.diff_part_from_asgn`.
    mempoolsync_params: Optional[Tuple[float, int, float]]
        When enabled, calculate the communication cost that MempoolSync would incur
        with the same amount of synchronizations. The first in the tuple is the
        parameter `Y`, the second is `DefTXtoSync`, the third is the 'much larger' factor.
        We consider a transaction pool to be much larger than `DefTXtoSync` when it is
        at least 'much larger' factor times larger.
    progress_fidelity: int
        Report progress of the main simulator loop each `progress_fidelity` events.
    _stats: Stats
        Statistics obtained from the simulation.
    _min_generation: int
        The generation of the youngest event currently in `eventq`.
    _eventq_buffer: List[Event]
        All events are first placed in `_eventq_buffer` and only
        later included into `eventq`.
    _d_matrix: np.array
        2D matrix of pair-wise differences counts
        (set only when `me_srep_dist_psi`).
    _s_vec: List[Set[int]]
        Initial set assignment vector (set only when `me_srep_dist_psi`).
    _s_sizes: Optional[List[int]]
        The exact sizes of sets. Either just calculated from an assignment
        when `me_srep_dist_psi` is a set assignment, or returned from
        `assign_diffs_from_set_sizes` when `me_srep_dist_psi` is a pair
        of set sizes distribution and psi.
    """

    network: Optional[nx.Graph] = None
    ws_nkp: Tuple[float, float, float] = (3, 2, 0.6)
    eventq: EventQ = field(default_factory=EventQ)
    timer: int = 0
    max_time: float = math.inf
    trace_file: Optional[str] = None
    me_srep_dist_psi: Optional[Union[DistUniSize_T, Assign_T]] = None
    me_srep_assume_complete: Optional[bool] = False
    mempoolsync_params: Optional[Tuple[float, int, float]] = None
    progress_fidelity: int = 1000
    _stats: Stats = field(default_factory=Stats)
    _min_generation: int = MIN_GENERATION_CONST
    _eventq_buffer: List[Event] = field(default_factory=list)
    _d_matrix: Optional[np.ndarray] = None
    _s_vec: Optional[List[Set[int]]] = None
    _s_sizes: Optional[List[int]] = None

    def __post_init__(self):
        """Complete initialization of simulator object."""
        # generate network topology
        if self.network is None:
            self.network = \
                nx.generators.random_graphs.connected_watts_strogatz_graph(
                    *self.ws_nkp)
        else:
            self.network = self.network.copy()

    def __initialize(self):
        """Initialize simulation."""
        if self.mempoolsync_params is not None:
            assert self.me_srep_dist_psi is not None, \
                "When MempoolSync is enabled, multi element SREP must be too."

        if self.me_srep_dist_psi is not None:
            p_len = len(self.me_srep_dist_psi)
            p_t = type(self.me_srep_dist_psi)
            if p_len > 2:
                self._s_vec = self.me_srep_dist_psi
                logging.info("Generating ME-SREP sets "
                             "directly from the passed assignment")
                self._s_sizes = [len(x) for x in self._s_vec]

                # calculate the mutual differences matrix
                self._d_matrix = aux.diff_part_from_asgn(
                    self.network,
                    self._s_vec,
                    assume_complete=self.me_srep_assume_complete)
            elif p_t == tuple and p_len == 2:
                dist, psi = self.me_srep_dist_psi
                logging.info("Generating ME-SREP sets assignment for network"
                             f" of {len(self.network.nodes)} nodes"
                             f" using provided distribution and psi={psi}.")
                self._s_vec, self._d_matrix, self._s_sizes \
                    = aux.assign_diffs_from_set_sizes(
                        dist=dist,
                        network=self.network,
                        psi=psi,
                        assume_complete=self.me_srep_assume_complete)
            else:
                raise ValueError(f"Unsupported {self.me_srep_dist_psi}")

            for n, data_set in zip(self.network.nodes, self._s_vec):
                node_data = NetworkNode(
                    idx=n,
                    data_set=data_set,
                    replicas={nbr: data_set.copy()
                              for nbr in self.network[n]})
                self.network.nodes[n]['data'] = node_data
        else:
            # initiate data sets and replicas for single element SREP
            for n in self.network.nodes:
                node_data = NetworkNode(
                    idx=n,
                    data_set=set([n]),
                    replicas={nbr: set([n])
                              for nbr in self.network[n]})
                self.network.nodes[n]['data'] = node_data

        # create the initial events
        initial_events = [NodeSyncEvent(node=n,
                                        current_time=0,
                                        generation=MIN_GENERATION_CONST,
                                        simulator=self,
                                        set_duration=True)
                          for n in self.network.nodes]
        self.eventq.enqueue(initial_events)

    def __is_fully_synchronized(self) -> bool:
        """Check whether full network synchronization is reached."""
        fst: Optional[Set[int]] = None

        for n in self.network.nodes:
            this: Set[int] = self.network.nodes[n]['data'].data_set
            if fst is None:
                fst = this
            elif fst != this:
                return False

        return True

    def calc_duration(self, node: int) -> int:
        """
        Calculate what would be the duration of `node`'s synchronization.

        Returns the duration of potential synchronization using the current
        state of the simulation.

        Parameters:
        --------
        node: int
            Node index in `self.network`.
        """
        G = self.network

        max_diff = 0
        for n in G.neighbors(node):
            this = G.nodes[node]['data'].data_set   # type: Set[int]
            neighbor = G.nodes[n]['data'].data_set  # type: Set[int]
            diffs = len(this.difference(neighbor)) \
                + len(neighbor.difference(this))
            max_diff = max(max_diff, diffs)

        return max_diff

    def calc_mempoolsync_cc(self, node: int):
        """
        Calculate what would be the communication cost of a MempoolSync.

        Parameters:
        --------
        node: int
            Node index in `self.network`.
        """
        y_const: float = self.mempoolsync_params[0]
        def_tx: int = self.mempoolsync_params[1]
        much_larger_factor = self.mempoolsync_params[2]
        mempool_size = self._s_sizes[node]

        if mempool_size < def_tx:
            return mempool_size
        elif mempool_size / def_tx >= much_larger_factor:
            return max(def_tx, int(y_const * mempool_size))
        else:
            return def_tx

    def __adjust_eventq(self):
        """
        Fill the `eventq` from `_eventq_buffer`.

        Make sure that all previous generation's `NodeSyncEvent`s have applied.
        Then adjust the duration of the `NodeSyncEvent`s in `_eventq_buffer`
        and enque them to `eventq`.
        """
        flush_eventq_buffer = False

        if not self.eventq.queue:
            flush_eventq_buffer = True
            min_gen = min(x.generation for x in self._eventq_buffer)
        else:
            # type: NodeSyncEvent
            min_gen = min(x.item.generation for x in self.eventq.queue)
            flush_eventq_buffer = min_gen > self._min_generation

        if flush_eventq_buffer:
            for i in range(len(self._eventq_buffer)):
                e = self._eventq_buffer[i]  # type: NodeSyncEvent
                e.adjust_duration()

            self.eventq.enqueue(self._eventq_buffer)
            self._eventq_buffer.clear()
            self._min_generation = min_gen

    def __trace(self, e: Event):
        """Trace one event."""
        graph_s = '\n'.join([f"{self.network.nodes[n]['data']}"
                             for n in self.network.nodes])
        s = ("----------------\n"
             f"Completed event:\n{e}\n"
             f"Entire graph:\n{graph_s}\n"
             f"Stats:\n{self.get_stats()}")

        if self.trace_file is not None:
            with open(self.trace_file, 'w') as f:
                f.write(s)
        else:
            print(s)

    def get_stats(self) -> Stats:
        """Return execution statistics."""
        self._stats.end_timer = self.timer
        return self._stats

    def __run(self) -> None:
        """Run the main simulator loop."""
        begin_t = time.time()
        self.__initialize()

        progress = 0
        while not self.eventq.empty() and self.timer < self.max_time:
            if self.__is_fully_synchronized():
                break

            event: Event = self.eventq.dequeue()
            new_events: List[Event] = event.apply()
            self._eventq_buffer += new_events
            self.timer = event.completion_time
            self.__adjust_eventq()

            self._stats.end_gen = max(self._stats.end_gen, event.generation)
            self.__trace(event)

            progress += 1
            if progress > 0 and progress % self.progress_fidelity == 0:
                logging.info(f"Progress: {progress} events passed.")

        self._stats.real_time = time.time() - begin_t

    def run(self, timeout: int = 10) -> None:
        """
        Run the main simulator loop.

        Send the alarm signal when the simulator takes too long.

        Parameters:
        --------
        timeout: int
            Timeout in seconds. 0 is interpreted as no timeout.
        """
        def handler(signum, frame):
            raise TimeoutError(f"{timeout} seconds timeout has passed.")

        # register the alarm signal
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        try:
            self.__run()
        except TimeoutError as e:
            # remember the graph that got simulator to timeout
            f_name = f"srep_timeout_graph_{uuid.uuid4().hex[:8]}.pickle"
            with open(f_name, 'wb') as f:
                pickle.dump(self.network, f)

            raise e

        # void the alarm signal
        signal.alarm(0)

@dataclass
class SREPSimulator_tvg():
    """
    The main simulator class for time varying graph.

    Attributes:
    ---------
    network: nx.Graph
        Topology.
    ws_nkp: Tuple[float, float, float]
        Parameters to
        `nx.generators.random_graphs.connected_watts_strogatz_graph`.
    eventq: EventQ
        Simulation event queue.
    timer: int
        Simulation timer.
    max_time: int
        Simulated time after which the simulation will get interrupted.
    trace_file: Optional[str]
        Path to log file that contains traces. If `None` print to stdout.
    me_srep_dist_psi: Optional[Union[DistUniSize_T, Assign_T]]
        When passed, multi elements SREP is initialized instead
        of the default single element SREP. `DistUniSize_T` is a tuple of
        `scipy.stats.rv_continuous` and `float` that represents the psi parameter.
        When an `Assign_T` (list of sets) is passed instead, it is used
        directly to initialize the data sets.
    me_srep_assume_complete: Optional[bool]
        When enabled, pass `assume_complete=True` in `aux.diff_part_from_asgn`.
    mempoolsync_params: Optional[Tuple[float, int, float]]
        When enabled, calculate the communication cost that MempoolSync would incur
        with the same amount of synchronizations. The first in the tuple is the
        parameter `Y`, the second is `DefTXtoSync`, the third is the 'much larger' factor.
        We consider a transaction pool to be much larger than `DefTXtoSync` when it is
        at least 'much larger' factor times larger.
    progress_fidelity: int
        Report progress of the main simulator loop each `progress_fidelity` events.
    _stats: Stats
        Statistics obtained from the simulation.
    _min_generation: int
        The generation of the youngest event currently in `eventq`.
    _eventq_buffer: List[Event]
        All events are first placed in `_eventq_buffer` and only
        later included into `eventq`.
    _d_matrix: np.array
        2D matrix of pair-wise differences counts
        (set only when `me_srep_dist_psi`).
    _s_vec: List[Set[int]]
        Initial set assignment vector (set only when `me_srep_dist_psi`).
    _s_sizes: Optional[List[int]]
        The exact sizes of sets. Either just calculated from an assignment
        when `me_srep_dist_psi` is a set assignment, or returned from
        `assign_diffs_from_set_sizes` when `me_srep_dist_psi` is a pair
        of set sizes distribution and psi.
    """

    network: Optional[nx.Graph] = None
    ws_nkp: Tuple[float, float, float] = (3, 2, 0.6)
    eventq: EventQ = field(default_factory=EventQ)
    timer: int = 0
    max_time: float = math.inf
    trace_file: Optional[str] = None
    me_srep_dist_psi: Optional[Union[DistUniSize_T, Assign_T]] = None
    me_srep_assume_complete: Optional[bool] = False
    mempoolsync_params: Optional[Tuple[float, int, float]] = None
    progress_fidelity: int = 1000
    _stats: Stats = field(default_factory=Stats)
    _min_generation: int = MIN_GENERATION_CONST
    _eventq_buffer: List[Event] = field(default_factory=list)
    _d_matrix: Optional[np.ndarray] = None
    _s_vec: Optional[List[Set[int]]] = None
    _s_sizes: Optional[List[int]] = None

    # time stamps when edge connection state changes
    time_stamp: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        """Complete initialization of simulator object."""
        # generate network topology
        if self.network is None:
            self.network, self.time_stamp = tvg.generate_tvg(self.ws_nkp)
            # print("Time Stamp:", self.time_stamp)
        else:
            self.network = self.network.copy()

    def __initialize(self):
        """Initialize simulation."""
        if self.mempoolsync_params is not None:
            assert self.me_srep_dist_psi is not None, \
                "When MempoolSync is enabled, multi element SREP must be too."

        if self.me_srep_dist_psi is not None:
            p_len = len(self.me_srep_dist_psi)
            p_t = type(self.me_srep_dist_psi)
            if p_len > 2:
                self._s_vec = self.me_srep_dist_psi
                logging.info("Generating ME-SREP sets "
                             "directly from the passed assignment")
                self._s_sizes = [len(x) for x in self._s_vec]

                # calculate the mutual differences matrix
                self._d_matrix = aux.diff_part_from_asgn(
                    self.network,
                    self._s_vec,
                    assume_complete=self.me_srep_assume_complete)
            elif p_t == tuple and p_len == 2:
                dist, psi = self.me_srep_dist_psi
                logging.info("Generating ME-SREP sets assignment for network"
                             f" of {len(self.network.nodes)} nodes"
                             f" using provided distribution and psi={psi}.")
                self._s_vec, self._d_matrix, self._s_sizes \
                    = aux.assign_diffs_from_set_sizes(
                        dist=dist,
                        network=self.network,
                        psi=psi,
                        assume_complete=self.me_srep_assume_complete)
            else:
                raise ValueError(f"Unsupported {self.me_srep_dist_psi}")

            for n, data_set in zip(self.network.nodes, self._s_vec):
                node_data = NetworkNode(
                    idx=n,
                    data_set=data_set,
                    replicas={nbr: data_set.copy()
                              for nbr in self.network[n]})
                self.network.nodes[n]['data'] = node_data
        else:
            # initiate data sets and replicas for single element SREP
            for n in self.network.nodes:
                node_data = NetworkNode(
                    idx=n,
                    data_set=set([n]),
                    replicas={nbr: set([n])
                              for nbr in self.network[n]})
                self.network.nodes[n]['data'] = node_data
                # print(node_data)

        # create the initial events
        initial_events = [NodeSyncEvent_tvg(node=n,
                                        current_time=0,
                                        generation=MIN_GENERATION_CONST,
                                        simulator=self,
                                        set_duration=True,
                                        time_stamp=self.time_stamp)
                        for n in self.network.nodes]
        self.eventq.enqueue(initial_events)

    def __is_fully_synchronized(self) -> bool:
        """Check whether full network synchronization is reached."""
        fst: Optional[Set[int]] = None

        for n in self.network.nodes:
            this: Set[int] = self.network.nodes[n]['data'].data_set
            # if fst is None:
            #     fst = this
            # elif fst != this:
            #     return False
            if len(this) != len(self.network.nodes):
                return False

        return True

    def calc_duration(self, node: int) -> int:
        """
        Calculate what would be the duration of `node`'s synchronization.

        Returns the duration of potential synchronization using the current
        state of the simulation.

        Parameters:
        --------
        node: int
            Node index in `self.network`.
        """
        G = self.network

        max_diff = 0
        for n in G.neighbors(node):
            this = G.nodes[node]['data'].data_set   # type: Set[int]
            neighbor = G.nodes[n]['data'].data_set  # type: Set[int]
            diffs = len(this.difference(neighbor)) \
                + len(neighbor.difference(this))
            max_diff = max(max_diff, diffs)

        return max_diff

    def calc_mempoolsync_cc(self, node: int):
        """
        Calculate what would be the communication cost of a MempoolSync.

        Parameters:
        --------
        node: int
            Node index in `self.network`.
        """
        y_const: float = self.mempoolsync_params[0]
        def_tx: int = self.mempoolsync_params[1]
        much_larger_factor = self.mempoolsync_params[2]
        mempool_size = self._s_sizes[node]

        if mempool_size < def_tx:
            return mempool_size
        elif mempool_size / def_tx >= much_larger_factor:
            return max(def_tx, int(y_const * mempool_size))
        else:
            return def_tx

    def __adjust_eventq(self):
        """
        Fill the `eventq` from `_eventq_buffer`.

        Make sure that all previous generation's `NodeSyncEvent`s have applied.
        Then adjust the duration of the `NodeSyncEvent`s in `_eventq_buffer`
        and enque them to `eventq`.
        """
        flush_eventq_buffer = False

        if not self.eventq.queue:
            flush_eventq_buffer = True
            min_gen = min(x.generation for x in self._eventq_buffer)
        else:
            # type: NodeSyncEvent
            min_gen = min(x.item.generation for x in self.eventq.queue)
            flush_eventq_buffer = min_gen > self._min_generation

        if flush_eventq_buffer:
            for i in range(len(self._eventq_buffer)):
                e = self._eventq_buffer[i]  # type: NodeSyncEvent
                # e.adjust_duration()

            self.eventq.enqueue(self._eventq_buffer)
            self._eventq_buffer.clear()
            self._min_generation = min_gen

    def __trace(self, e: Event):
        """Trace one event."""
        graph_s = '\n'.join([f"{self.network.nodes[n]['data']}"
                             for n in self.network.nodes])
        s = ("----------------\n"
             f"Completed event:\n{e}\n"
             f"Entire graph:\n{graph_s}\n"
             f"Stats:\n{self.get_stats()}")

        if self.trace_file is not None:
            with open(self.trace_file, 'w') as f:
                f.write(s)
        else:
            print(s)

    def get_stats(self) -> Stats:
        """Return execution statistics."""
        self._stats.end_timer = max(self.timer, self._stats.end_timer)
        return self._stats

    def __run(self) -> None:
        """Run the main simulator loop."""
        begin_t = time.time()
        self.__initialize()

        progress = 0
        while not self.eventq.empty() and self.timer < self.max_time:
            if self.__is_fully_synchronized():
                break

            event: Event = self.eventq.dequeue()
            new_events: List[Event] = event.apply()
            self._eventq_buffer += new_events
            self.timer = event.completion_time
            # print("timer:", self.timer)
            self.__adjust_eventq()
            self.get_stats()
            # print("Generation:", self._min_generation)

            self._stats.end_gen = max(self._stats.end_gen, event.generation)
            self.__trace(event)

            progress += 1
            if progress > 0 and progress % self.progress_fidelity == 0:
                logging.info(f"Progress: {progress} events passed.")

        self._stats.real_time = time.time() - begin_t
        # print("Stat end timer:", self._stats.end_timer)

    def run(self, timeout: int = 10) -> None:
        """
        Run the main simulator loop.

        Send the alarm signal when the simulator takes too long.

        Parameters:
        --------
        timeout: int
            Timeout in seconds. 0 is interpreted as no timeout.
        """
        def handler(signum, frame):
            raise TimeoutError(f"{timeout} seconds timeout has passed.")

        # register the alarm signal
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        try:
            self.__run()
        except TimeoutError as e:
            # remember the graph that got simulator to timeout
            f_name = f"srep_timeout_graph_{uuid.uuid4().hex[:8]}.pickle"
            with open(f_name, 'wb') as f:
                pickle.dump(self.network, f)

            raise e

        # void the alarm signal
        signal.alarm(0)

def me_srep_analytic(network: nx.Graph, asgn: Assign_T) -> Stats:
    """
    Solve ME-SREP analytically.

    Parameters:
    --------
    network: nx.Graph
        Network as a graph.
    asgn: Assign_T
        Initial set assignment.
    """
    n = len(network.nodes)

    assert nx.is_connected(network)
    assert len(asgn) == n, \
        f"Network of {n} nodes, but {len(asgn)} assignments."

    begin_t = time.time()

    # convert the assignment to `np.ndarray`. Python's list of sets of
    # integers takes ~5x more memory than the equivalent `np.ndarray`.
    if not isinstance(asgn, np.ndarray):
        asgn_np: np.ndarray = np.empty(n, dtype=object)
        for i, s in enumerate(asgn):
            asgn_np[i] = np.array(list(s), dtype=np.int64)

        asgn = asgn_np

    M: np.ndarray = aux.mut_diff_from_asgn(network, asgn, use_np=True)
    # rounds -1 for compatibility with `SREPSimulator`
    rounds, communication = -1, 0
    while M.sum():
        new_asgn: np.ndarray = np.empty(n, dtype=object)
        for i in tqdm.tqdm(range(n)):
            new_asgn[i] = asgn[i]
            for j in network[i]:
                new_asgn[i] = np.union1d(new_asgn[i], asgn[j])

        communication += M.sum()
        rounds += 1

        asgn = new_asgn
        M = aux.mut_diff_from_asgn(network, asgn, use_np=True)

    return Stats(communication_cost=communication,
                 end_gen=rounds,
                 real_time = time.time() - begin_t,
                 redundant_trans=-math.inf,
                 sync_invocations=-math.inf)
