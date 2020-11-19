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


def fit(opf, X_train, Y_train, I_train=None):
    """Fits data in the classifier.

    Args:
        X_train (np.array): Array of training features.
        Y_train (np.array): Array of training labels.
        I_train (np.array): Array of training indexes.

    """

    logger.info('Fitting classifier ...')

    # Initializing the timer

    # Creating a subgraph
    opf.subgraph = Subgraph(X_train, Y_train, I=I_train)

    # Finding prototypes
    opf._find_prototypes()
    start = time.time()

    # Creating a minimum heap
    h = Heap(size=opf.subgraph.n_nodes)

    # For each possible node
    for i in range(opf.subgraph.n_nodes):
        # Checks if node is a prototype
        if opf.subgraph.nodes[i].status == c.PROTOTYPE:
            # If yes, it does not have predecessor nodes
            opf.subgraph.nodes[i].pred = c.NIL

            # Its predicted label is the same as its true label
            opf.subgraph.nodes[i].predicted_label = opf.subgraph.nodes[i].label

            # Its cost equals to zero
            h.cost[i] = 0

            # Inserts the node into the heap
            h.insert(i)

        # If node is not a prototype
        else:
            # Its cost equals to maximum possible value
            h.cost[i] = c.FLOAT_MAX

    # While the heap is not empty
    while not h.is_empty():

        # Removes a node
        p = h.remove()

        # Appends its index to the ordered list
        opf.subgraph.idx_nodes.append(p)

        # Gathers its cost
        opf.subgraph.nodes[p].cost = h.cost[p]

        # For every possible node
        for q in range(opf.subgraph.n_nodes):
            # If we are dealing with different nodes
            if p != q:
                # If `p` node cost is smaller than `q` node cost
                if h.cost[p] < h.cost[q]:
                    # Checks if we are using a pre-computed distance
                    if opf.pre_computed_distance:
                        # Gathers the distance from the distance's matrix
                        weight = opf.pre_distances[opf.subgraph.nodes[p].idx][opf.subgraph.nodes[q].idx]

                    # If the distance is supposed to be calculated
                    else:
                        # Calls the corresponding distance function
                        weight = opf.distance_fn(
                            opf.subgraph.nodes[p].features, opf.subgraph.nodes[q].features)

                    # The current cost will be the maximum cost between the node's and its weight (arc)
                    current_cost = np.maximum(h.cost[p], weight)

                    # If current cost is smaller than `q` node's cost
                    if current_cost < h.cost[q]:
                        # `q` node has `p` as its predecessor
                        opf.subgraph.nodes[q].pred = p

                        # And its predicted label is the same as `p`
                        opf.subgraph.nodes[q].predicted_label = opf.subgraph.nodes[p].predicted_label

                        # Updates the heap `q` node and the current cost
                        h.update(q, current_cost)

    # The subgraph has been properly trained
    opf.subgraph.trained = True

    # Ending timer
    end = time.time()

    # Calculating training task time
    train_time = end - start

    logger.info('Classifier has been fitted.')
    logger.info('Training time: %s seconds.', train_time)