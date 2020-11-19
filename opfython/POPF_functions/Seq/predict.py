import time
import numpy as np
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import Subgraph


logger = log.get_logger(__name__)

def predict(opf, X_val, I_val=None):
    """Predicts new data using the pre-trained classifier.

    Args:
        X_val (np.array): Array of validation or test features.
        I_val (np.array): Array of validation or test indexes.

    Returns:
        A list of predictions for each record of the data.

    """

    # Checks if there is a subgraph
    if not opf.subgraph:
        # If not, raises an BuildError
        raise e.BuildError('Subgraph has not been properly created')

    # Checks if subgraph has been properly trained
    if not opf.subgraph.trained:
        # If not, raises an BuildError
        raise e.BuildError('Classifier has not been properly fitted')

    logger.info('Predicting data ...')

    # Initializing the timer
    start = time.time()

    # Creating a prediction subgraph
    pred_subgraph = Subgraph(X_val, I=I_val)

    # For every possible node
    for i in range(pred_subgraph.n_nodes):
        # Initializing the conqueror node
        conqueror = -1

        # Initializes the `j` counter
        j = 0

        # Gathers the first node from the ordered list
        k = opf.subgraph.idx_nodes[j]

        # Checks if we are using a pre-computed distance
        if opf.pre_computed_distance:
            # Gathers the distance from the distance's matrix
            weight = opf.pre_distances[opf.subgraph.nodes[k].idx][pred_subgraph.nodes[i].idx]

        # If the distance is supposed to be calculated
        else:
            # Calls the corresponding distance function
            weight = opf.distance_fn(
                opf.subgraph.nodes[k].features, pred_subgraph.nodes[i].features)

        # The minimum cost will be the maximum between the `k` node cost and its weight (arc)
        min_cost = np.maximum(opf.subgraph.nodes[k].cost, weight)

        # The current label will be `k` node's predicted label
        current_label = opf.subgraph.nodes[k].predicted_label

        # While `j` is a possible node and the minimum cost is bigger than the current node's cost
        while j < (opf.subgraph.n_nodes - 1) and min_cost > opf.subgraph.nodes[opf.subgraph.idx_nodes[j + 1]].cost:
            # Gathers the next node from the ordered list
            l = opf.subgraph.idx_nodes[j + 1]

            # Checks if we are using a pre-computed distance
            if opf.pre_computed_distance:
                # Gathers the distance from the distance's matrix
                weight = opf.pre_distances[opf.subgraph.nodes[l].idx][pred_subgraph.nodes[i].idx]

            # If the distance is supposed to be calculated
            else:
                # Calls the corresponding distance function
                weight = opf.distance_fn(
                    opf.subgraph.nodes[l].features, pred_subgraph.nodes[i].features)

            # The temporary minimum cost will be the maximum between the `l` node cost and its weight (arc)
            temp_min_cost = np.maximum(opf.subgraph.nodes[l].cost, weight)

            # If temporary minimum cost is smaller than the minimum cost
            if temp_min_cost < min_cost:
                # Replaces the minimum cost
                min_cost = temp_min_cost

                # Gathers the identifier of `l` node
                conqueror = l

                # Updates the current label as `l` node's predicted label
                current_label = opf.subgraph.nodes[l].predicted_label

            # Increments the `j` counter
            j += 1

            # Makes `k` and `l` equals
            k = l

        # Node's `i` predicted label is the same as current label
        pred_subgraph.nodes[i].predicted_label = current_label

        # Checks if any node has been conquered
        if conqueror > -1:
            # Marks the conqueror node and its path
            opf.subgraph.mark_nodes(conqueror)

    # Creating the list of predictions
    preds = [pred.predicted_label for pred in pred_subgraph.nodes]

    # Ending timer
    end = time.time()

    # Calculating prediction task time
    predict_time = end - start

    logger.info('Data has been predicted.')
    logger.info('Prediction time: %s seconds.', predict_time)

    return preds