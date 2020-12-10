from multiprocessing import JoinableQueue, Queue,Array
import opfython.utils.logging as log
import opfython.utils.exception as e
from opfython.POPF_functions.POPF_functions import creaTagli,creaProcFit
from opfython.core import  Subgraph
import numpy as np
import time


logger = log.get_logger(__name__)

# Parte cocorrente per il preict
def predConc(opf, work, X_val, result, conquerors):
    while True:
        # predno il range su cui fare il predict

        ran = work.get()
        # faccio il predict su un range di X_val

        pred = predict(opf,X_val[ran[0]:ran[1]],conquerors)

        j = 0
        # mi salvo i pred nell'ordine giusto
        for i in range(ran[0], ran[1]):

            result[i] = pred[j]
            j += 1

        work.task_done()


def pred(opf, X_val,I_val=None):
    logger.info('Predicting data ...')


    if(len(X_val)<500):
        return predict(opf,X_val,I_val=I_val)

    # Initializing the timer
    start = time.time()
    # tagli
    t = []
    # processi
    p = []


    creaTagli(opf._tagli, t, len(X_val))

    work = JoinableQueue()

    # Conquerors non sono altro che i nodi che hanno conquistato i nodi di X_val su cui faremo mark_nodes() per settare i nodi Rilevanti e quelli non Rilevanti
    conquerors = Queue()

    # ci vanno i risultati in ordine giusto
    result = Array('i', len(X_val), lock=False)

    creaProcFit(predConc, p, opf._processi,opf, work, X_val, result, conquerors)

    # Do il lavoro ai processi
    for i in range(len(t)):
        work.put(t[i])

    # Aspetto che terminano
    work.join()

    # termino i processi (Utile per non intasare la memoria con processi quando si usa il pruring)
    for i in range(opf._processi):
        p[i].terminate()



    # Ending timer
    end = time.time()

    # Calculating prediction task time
    predict_time = end - start

    logger.info('Data has been predicted.')
    logger.info('Prediction time: %s seconds.', predict_time)

    return result,conquerors




def predict(opf, X_val, coda, I_val=None):
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


    # Initializing the timer
    start = time.time()

    # Creating a prediction subgraph
    pred_subgraph = Subgraph(X_val, I=I_val)

    startPredict(opf,pred_subgraph,coda)

    # Creating the list of predictions
    preds = [pred.predicted_label for pred in pred_subgraph.nodes]

    if coda==None:
        # Ending timer
        end = time.time()

        # Calculating prediction task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info('Prediction time: %s seconds.', predict_time)

    return preds


def startPredict(opf,pred_subgraph,coda):
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
            coda.put(conqueror)
