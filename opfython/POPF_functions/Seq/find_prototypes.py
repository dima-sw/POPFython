import copy
import time

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import OPF, Heap, Subgraph


logger = log.get_logger(__name__)


def _find_prototypes(opf):
    """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

    """

    logger.debug('Finding prototypes ...')

    start = time.time()

    # Creating a Heap of size equals to number of nodes
    h = Heap(opf.subgraph.n_nodes)

    # Marking first node without any predecessor
    opf.subgraph.nodes[0].pred = c.NIL

    # Adding first node to the heap
    h.insert(0)

    # Creating a list of prototype nodes
    prototypes = []

    # While the heap is not empty
    while not h.is_empty():
        # Remove a node from the heap
        p = h.remove()

        # Gathers its cost from the heap
        opf.subgraph.nodes[p].cost = h.cost[p]

        # And also its predecessor
        pred = opf.subgraph.nodes[p].pred

        # If the predecessor is not NIL
        if pred != c.NIL:
            # Checks if the label of current node is the same as its predecessor
            if opf.subgraph.nodes[p].label != opf.subgraph.nodes[pred].label:
                # If current node is not a prototype
                if opf.subgraph.nodes[p].status != c.PROTOTYPE:
                    # Marks it as a prototype
                    opf.subgraph.nodes[p].status = c.PROTOTYPE

                    # Appends current node identifier to the prototype's list
                    prototypes.append(p)

                # If predecessor node is not a prototype
                if opf.subgraph.nodes[pred].status != c.PROTOTYPE:
                    # Marks it as a protoype
                    opf.subgraph.nodes[pred].status = c.PROTOTYPE

                    # Appends predecessor node identifier to the prototype's list
                    prototypes.append(pred)

        # For every possible node
        for q in range(opf.subgraph.n_nodes):
            # Checks if the color of current node in the heap is not black
            if h.color[q] != c.BLACK:
                # If `p` and `q` identifiers are different
                if p != q:
                    # If it is supposed to use pre-computed distances
                    if opf.pre_computed_distance:
                        # Gathers the arc from the distances' matrix
                        weight = opf.pre_distances[opf.subgraph.nodes[p].idx][opf.subgraph.nodes[q].idx]

                    # If distance is supposed to be calculated
                    else:
                        # Calculates the distance
                        weight = opf.distance_fn(
                            opf.subgraph.nodes[p].features, opf.subgraph.nodes[q].features)

                    # If current arc's cost is smaller than the path's cost
                    if weight < h.cost[q]:
                        # Marks `q` predecessor node as `p`
                        opf.subgraph.nodes[q].pred = p

                        # Updates the arc on the heap
                        h.update(q, weight)
    end = time.time()
    fittime = end - start
    logger.debug('Prototypes: %s.', prototypes)
    logger.info('Prototypes found in: %s seconds.', fittime)