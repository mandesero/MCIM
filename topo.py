import typing as T
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from time import time

import metis
import CTCEHC_mst


@dataclass
class Cluster:
    vertices: list = field(default_factory=list)
    edges: list = field(default_factory=list)


class TopoGraph:
    def __init__(self, graph: T.Optional[nx.Graph] = None):
        self.nx_graph = graph or nx.Graph()
        self.clusters: dict[int, Cluster] = []
        self.tunnels: list[tuple[int, int]] = []
        # cluster ids of nodes in order of nx_graph.nodes
        self.parts: list[int] = []

    def define_leaves(self) -> list:
        """
        Create list of idents leave-nodes in graph.

        :return: List of idents leave-nodes.
        """
        return [key for key, val in nx.degree(self.nx_graph) if val == 1]

    def split_graph(self, n: int, part_graph: callable = metis.part_graph, params: dict = {}) -> None:
        """
        Splits the network graph into 'n' sub-graphs using the METIS library.

        :param n: The number of parts to split the graph into.
        """
        if n < 1:
            raise ValueError("Number of parts 'n' must be at least 1.")
        t0 = time()
        match part_graph:
            case metis.part_graph:
                _, self.parts = metis.part_graph(
                    self.nx_graph, n, contig=True, compress=True, **params
                )
            case CTCEHC_mst.CTCEHC_part:
                self.parts = CTCEHC_mst.CTCEHC_part(self.nx_graph, n)
            case _:
                return
            
        node_ids = dict(enumerate(self.nx_graph.nodes()))
        node_to_id = {v: k for k, v in node_ids.items()}

        # create and fill clusters with vertices
        self.clusters = {p: Cluster() for p in set(self.parts)}
        for i, p in enumerate(self.parts):
            self.clusters[p].vertices.append(node_ids[i])

        # form tunnels and add links to clusters
        self.tunnels = []
        for edge in self.nx_graph.edges():
            src, dst = edge
            src_p = self.parts[node_to_id[src]]
            dst_p = self.parts[node_to_id[dst]]
            if src_p == dst_p:
                self.clusters[src_p].edges.append(edge)
            else:
                self.tunnels.append(edge)

        return self.parts, time() - t0


    def get_cluster_graph(self) -> nx.Graph:
        """
        Constructs a cluster graph based on the partitioning of the original graph.

        :return: NetworkX graph
        """
        res_edges = defaultdict(int)  # (cluster 1 id, cluster 2 id) -> bw
        node_to_part = dict(zip(self.nx_graph.nodes, self.parts))
        for edge in self.tunnels:
            v, u = edge[0], edge[1]
            res_e = tuple(sorted([node_to_part[v], node_to_part[u]]))
            bw = self.nx_graph.get_edge_data(u, v)["bw"]
            res_edges[res_e] += bw

        res_g = nx.Graph()
        for edge, bw in res_edges.items():
            res_g.add_edge(edge[0], edge[1], bw=bw)

        return res_g
