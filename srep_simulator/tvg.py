"""
Functions for time-varying graph extension

Author: Xingyu Chen <chxy517@bu.edu>
Date: August, 2023.
"""

from typing import List, Set, Tuple, Optional, Generator

import networkx as nx
import numpy as np
import random 

def generate_tvg(ws_nkp: Tuple[float, float, float]) -> Tuple[nx.Graph, np.ndarray]:
    """
    Generate n-nodes time-varying graph

    Parameters:
    --------
    ws_nkp: Tuple[float, float, float]
        set of parameters
    """
    # Create the graph
    net_size = ws_nkp[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(net_size))
    while not nx.is_connected(graph):
        u, v = random.sample(graph.nodes(), 2)
        graph.add_edge(u, v)

    edges = graph.edges()

    prob_con = 0.05
    prob_discon = 1 - prob_con

    stamp_arr = {}

    for edge in edges:
        bool_array = np.random.choice([True, False], size=1000, p=[prob_con, prob_discon])
        array = []
        for index in range(0, 1000):
            if bool_array[index]:
                array.append(index)
        stamp_arr[edge] = array 
    
    return graph, stamp_arr


def generate_tvl(ws_nkp: Tuple[float, float, float]) -> Tuple[nx.Graph, np.ndarray]:
    """
    Generate time-varying line

    Uses Nauty. Calls it in a separate process.

    Parameters:
    --------
    ws_nkp: Tuple[float, float, float]
        set of parameters
    """
    net_size = ws_nkp[0]

    # Create the graph
    graph = nx.Graph()
    graph.add_nodes_from(list(range(net_size)))

    prob_con = 0.05
    prob_discon = 1 - prob_con

    stamp_arr = []
    for i in range (net_size - 1):
        bool_array = np.random.choice([True, False], size=1000, p=[prob_con, prob_discon])
        array = []
        for index in range(0, 1000):
            if bool_array[index]:
                array.append(index)
        stamp_arr.append(array)
    
    return graph, stamp_arr

def check_connection_exp(array, time):
    # Check if the connection state of an edge needs to be changed
    # Return true if change needed, false if stays same
    sorted_array = sorted(array)
    # Find the two indexes where time falls in between
    # Apply binary search
    left_index = 0
    right_index = len(sorted_array) - 1
    while left_index <= right_index:
        mid_index = (left_index + right_index) // 2
        mid_value = sorted_array[mid_index]

        if mid_value == time:
            return mid_index, mid_index + 1
        elif mid_value < time:
            left_index = mid_index + 1
        else:
            right_index = mid_index - 1
    if right_index % 2 == 1:
        return False
    else:
        return True

def check_connection(array, time):
    if time in array:
        return True
    else:
        return False


def update_graph(graph, time_stamp, current_time) -> nx.Graph:
    for edge, time_arr in  time_stamp.items():
        node1 = edge[0]
        node2 = edge[1]
        if check_connection(time_arr, current_time):
            graph.add_edge(node1, node2)
            if node2 not in graph.nodes[node1]['data'].replicas:
                graph.nodes[node1]['data'].replicas[node2] = {node1}
            if node1 not in graph.nodes[node2]['data'].replicas:
                graph.nodes[node2]['data'].replicas[node1] = {node2}
            # print("Edge added: ", i, i + 1, "time: ", current_time)
        else:
            if graph.has_edge(node1, node2):
                graph.remove_edge(node1, node2)
                # print("Edge removed: ", i, i + 1, "time: ", current_time)
    return graph
    

def generate_graph_with_diameter(diameter):
    graph = nx.random_tree(20)  # Generate a random tree with diameter + 1 nodes
    while nx.diameter(graph) != diameter:
        graph = nx.random_tree(20)
    
    return graph