import random
from copy import copy
from itertools import permutations
from typing import Generator
from collections import defaultdict

import networkx as nx
import numpy as np
from time import time
from functools import lru_cache
from crossover import *


def random_permutations(L: list, count: int) -> Generator[list, None, None]:
    """
    Generates and yields a specified number of random permutations of a list.

    :param L: The original list to permute.
    :param count: The number of random permutations to generate.
    :return: A generator yielding random permutations of the list.
    """
    shuffle_list = copy(L)
    for _ in range(count):
        random.shuffle(shuffle_list)
        yield shuffle_list


def get_diff(G1: nx.Graph, G2: nx.Graph, m_map: dict) -> tuple[int, int]:
    """
    Calculates the total difference and loss in bandwidth between matched edges in G1 and G2.

    :param m_map: A mapping of nodes from G1 to G2.
    :return: A tuple containing the total difference and total loss in bandwidth.
    """
    K1 = float("inf")
    K2 = float("inf")
    for u, v in G1.edges:
        x = m_map[u]
        y = m_map[v]
        bw_1 = G1.get_edge_data(u, v)["bw"]
        bw_2 = G2.get_edge_data(x, y)["bw"] if G2.has_edge(x, y) else 0
        diff = bw_1 - bw_2
        if bw_2 != 0:
            K1 = min(K1, diff)
        K2 = min(K2, diff)
    return K1, K2

@lru_cache
def find_path_with_max_bw(G: nx.Graph, source, target):
    """
    """

    if G.has_edge(source, target):
        return [source, target], G.get_edge_data(source, target)["bw"]

    predecessors = {node: None for node in G.nodes}

    max_bw_to_node = {node: float('-inf') for node in G.nodes}
    max_bw_to_node[source] = float('inf')

    queue = [(source, float('inf'))]

    while queue:
        current_node, current_bw = queue.pop(0)

        for neighbor in G.neighbors(current_node):
            edge_bw = G[current_node][neighbor]['bw']
            new_bw = min(current_bw, edge_bw)

            if new_bw > max_bw_to_node[neighbor]:
                max_bw_to_node[neighbor] = new_bw
                predecessors[neighbor] = current_node
                queue.append((neighbor, new_bw))

    path = []
    current_node = target
    while current_node != source:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.append(source)
    path.reverse()

    return path, max_bw_to_node[target]


def K_MIN(G1: nx.Graph, G2: nx.Graph, mmap: dict):
    """
    """
    F = {}
    for u, v in G1.edges:
        bw_1 = G1.get_edge_data(u, v)["bw"]

        x, y = mmap[u], mmap[v]
        path, tmp = find_path_with_max_bw(G2, x, y)
        F[(u, v)] = path
    
    S = defaultdict(int)
    
    for link, path in F.items():
        for i in range(len(path) - 1):
            x, y = sorted(path[i:i+2])
            S[(x, y)] += G1.get_edge_data(link[0], link[1])["bw"]
    return max(value - G2.get_edge_data(key[0], key[1])["bw"] for key, value in S.items())


def random_matching(G1: nx.Graph, G2: nx.Graph, it: int =100000) -> dict:
    """
    Finds a random matching between nodes of two graphs that minimizes the difference in bandwidth.

    :param G1: The first graph.
    :param G2: The second graph.
    :param it: The number of iterations for random permutations, used when the number of nodes is large.
    :return: A dictionary representing the best matching found.
    """

    best = None
    
    K = float("inf")

    nodes_1 = list(G1.nodes)
    nodes_2 = list(G2.nodes)

    if len(nodes_1) != len(nodes_2):
        raise

    # if len(nodes_1) >= 10:
    #     iterator = random_permutations(nodes_2, it)
    # else:
    iterator = permutations(nodes_2)

    for n2 in iterator:
        mmap = dict(zip(nodes_1, n2))
        new_K = K_MIN(G1, G2, mmap)
        if new_K < K:
            K = new_K
            best = mmap
    return best

def mapping(matrix):
    d = {}
    ignored_columns = set()
    for row_index, row in enumerate(matrix):
        mask = np.ones(row.shape, dtype=bool)
        mask[list(ignored_columns)] = False
        
        masked_row = np.where(mask, row, -np.inf)
        max_index = np.argmax(masked_row)
        d[row_index] = max_index
        ignored_columns.add(max_index)
    return d


def power_iteration(A1, A2, alpha=0.5, max_iter=100, tol=1e-6):
    n1, n2 = A1.shape[0], A2.shape[0]
    
    R = np.ones((n1, n2)) / (n1 * n2)
    
    D1_inv = np.diag(1 / A1.sum(axis=1).flatten()) 
    D2_inv = np.diag(1 / A2.sum(axis=1).flatten())
    P1 = A1.dot(D1_inv)
    P2 = A2.dot(D2_inv)
    P = np.kron(P1, P2)
    
    for _ in range(max_iter):
        R_new = (1 - alpha) * P.dot(R.flatten()).reshape(n1, n2) + alpha * R
        R_new /= np.linalg.norm(R_new, ord=1)
        
        if np.linalg.norm(R_new - R, ord=1) < tol:
            break
        
        R = R_new
    
    return R


def isorank(G1, G2, num_iterations=10, alpha=0.85, tol=1e-6):
    A1 = nx.adjacency_matrix(G1).todense()
    A2 = nx.adjacency_matrix(G2).todense()
    
    R_iterative = power_iteration(A1, A2, alpha, num_iterations, tol)
    
    return mapping(R_iterative)

# =================================================================================================================================

def initialize_population(G1, G2, population_size):
    population = []
    for _ in range(population_size):
        alignment = np.random.permutation(G2.number_of_nodes())
        population.append(alignment)
    return population


def mutate(alignment, mutation_rate):
    for i in range(len(alignment)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(alignment) - 1)
            alignment[i], alignment[j] = alignment[j], alignment[i]
    return alignment

def fitness_function_degree(alignment, G1, G2):
    score = 0
    for i, node1 in enumerate(G1.nodes()):
        for j, node2 in enumerate(G1.nodes()):
            if G1.has_edge(node1, node2) and G2.has_edge(alignment[i], alignment[j]):
                score += 1
    return score


def fitness_function_k_min(m_map, G1: nx.Graph, G2: nx.Graph):
    return K_MIN(G1, G2, m_map)
        
start_population = {}

def magna_plus_plus(G1, G2, fitness: callable, crossover: callable, population_size=25, generations=100, mutation_rate=0.02):
    ks = []

    if (G1, G2) not in start_population:
        start_population[(G1, G2)] = initialize_population(G1, G2, population_size)

    population = start_population[(G1, G2)][:]

    if fitness == fitness_function_k_min:
        population.sort(key=lambda x: fitness(x, G1, G2))
    elif fitness == fitness_function_degree:
        population.sort(key=lambda x: fitness(x, G1, G2), reverse=True)

    ks.append(K_MIN(G1, G2, population[0]))

    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            if crossover == Crossover.VR:
                parents = [population[0], *random.sample(population, 3)]
            else:
                parents = [population[0], *random.sample(population, 1)]
                           
            child1, child2 = crossover(*parents)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

            parents = random.sample(population, 2)
            child1, child2 = crossover(*parents)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        if fitness == fitness_function_k_min:
            new_population.sort(key=lambda x: fitness(x, G1, G2))
        elif fitness == fitness_function_degree:
            new_population.sort(key=lambda x: fitness(x, G1, G2), reverse=True)

        population = new_population[:population_size]
        ks.append(K_MIN(G1, G2, population[0]))
    
    return min(population, key=lambda x: fitness_function_k_min(x, G1, G2)), ks

# =================================================================================================================================

def initialize_similarity(G1, G2):
    """
    Инициализация сходства на основе степеней вершин.
    """
    similarity = np.zeros((len(G1.nodes()), len(G2.nodes())))
    for i, node1 in enumerate(G1.nodes()):
        for j, node2 in enumerate(G2.nodes()):
            similarity[i, j] = 1 / (1 + abs(G1.degree(node1) - G2.degree(node2)))
    return similarity

def update_similarity(G1, G2, similarity):
    """
    Обновление сходства на основе сходства соседей.
    """
    new_similarity = np.zeros_like(similarity)
    for i, node1 in enumerate(G1.nodes()):
        for j, node2 in enumerate(G2.nodes()):
            # Суммируем сходство всех соседей node1 с каждым соседом node2
            sum_sim = 0
            for neighbor1 in G1.neighbors(node1):
                for neighbor2 in G2.neighbors(node2):
                    i_neighbor = list(G1.nodes()).index(neighbor1)
                    j_neighbor = list(G2.nodes()).index(neighbor2)
                    sum_sim += similarity[i_neighbor, j_neighbor]
            new_similarity[i, j] = sum_sim
    max_sim = np.max(new_similarity)
    if max_sim > 0:
        new_similarity /= max_sim
    return new_similarity

def netal_enhanced(G1, G2, iterations=100):
    """
    Расширенная версия алгоритма NETAL.
    """
    similarity = initialize_similarity(G1, G2)
    for _ in range(iterations):
        similarity = update_similarity(G1, G2, similarity)
    return mapping(similarity)

import networkx.algorithms.isomorphism.vf2pp as vf2pp
def vf2(G1, G2):
    return vf2pp.vf2pp_isomorphism(G1, G2)

import vf3py
def vf3(G1, G2):
    m = vf3py.get_exact_isomorphisms(G1, G2)
    if m:
        return m[0]
    return None


def matching_graphs(G1: nx.Graph, G2: nx.Graph, match: callable, params: dict={}):
    ks = None
    t0 = time()
    if match == magna_plus_plus:
        m, ks = match(G1, G2, **params)
    else:
        m = match(G1, G2, **params)
    if m is None:
        return None
    return m, time() - t0, ks

