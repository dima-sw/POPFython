import numpy as np
import opfython.utils.constants as c
import opfython.utils.logging as log
import opfython.math.general as g

logger = log.get_logger(__name__)


def prune(opf, X_train, Y_train, X_val, Y_val,M_loss, n_iterations=10):
    """Prunes a classifier over a validation set.
    Args:
        X_train (np.array): Array of training features.
        Y_train (np.array): Array of training labels.
        X_val (np.array): Array of validation features.
        Y_val (np.array): Array of validation labels.
        n_iterations (int): Maximum number of iterations of learning.

    """

    logger.info('Pruning classifier ...')

    # Faccio il primo learning e mi calcolo l'accuratezza massima
    opf.learn(X_train, Y_train, X_val, Y_val,n_iterations=n_iterations)

    """Penso Sia ridondante perche' gia' lo facciamo nel learning"""
    preds = opf.pred(X_val)
    # Calculating accuracy
    acc = g.opf_accuracy(Y_val, preds)

    # Prendo il numero di nodi iniziali del grafo
    initial_nodes = opf.subgraph.n_nodes

    # Faccio partire il pruring
    pruringRun(opf,acc, M_loss,n_iterations, X_train, Y_train, X_val, Y_val)

    # Prendo i nodi a fine pruring
    final_nodes = opf.subgraph.n_nodes
    logger.info('Initial number of nodes: %s , final number of nodes: %s,', initial_nodes, final_nodes)
    # Calculating pruning ratio
    prune_ratio = 1 - final_nodes / initial_nodes

    logger.info('Prune ratio: %s.', prune_ratio)


def pruringRun(opf, acc, M_loss,n_iterations, X_train, Y_train, X_val, Y_val):
    # tmp= accuratezza attuale
    tmp = acc
    # flag serve per tenere traccia se ci sia ancora almeno un nodo non rilevante
    flag=True

    # mentre l'accuratezza attuale è >= (dell'accuratezza massima iniziale - M_loss) e mentre ci sta almeno un nodo non Rilevante (flag)
    while abs(acc - tmp) <= M_loss and flag:
        flag=False
        #Lunghezza prima di rimuovere i nodi irrilevanti
        lunIn=len(X_train)

        # Rimuovo i nodi irrillevanti
        #I nodi irrellevanti li rimuovi da X_train e Y_train e li aggiungo a X_val e Y_val
        X_train, Y_train, X_val, Y_val = pruringUpdateList(opf.subgraph.nodes, X_train, Y_train, X_val, Y_val)

        #Lunghezza dopo aver tolto i nodi irrilevanti
        lunFin=len(X_train)
        #Se la lunghezza non cambia significa che possiamo terminare il pruring, altrimenti facciamo partire un nuovo learning
        if lunFin<lunIn:
            flag=True
            # Faccio il learning e prendo l'accuratezza
            opf.learn(X_train, Y_train, X_val, Y_val,n_iterations=n_iterations)
            """Penso Sia ridondante perche' gia' lo facciamo nel learning"""
            preds = opf.pred(X_val)
            # Calculating accuracy
            tmp = g.opf_accuracy(Y_val, preds)
            logger.info('Current accuracy: %s.', tmp)


def pruringUpdateList(nodes,X_train, Y_train, X_val, Y_val):
    # Liste temporanee per X_train, Y_trrain, X_val e Y_val
    X_temp, Y_temp = [], []
    X_t, Y_t = [], []


    for j, n in enumerate(nodes):
        # Aggiungo alle liste temporanee X_temp, Y_temp i nodi rilevanti
        if n.relevant != c.IRRELEVANT:
            X_temp.append(X_train[j, :])
            Y_temp.append(Y_train[j])
        # Aggiungo alle liste temporanee X_t e Y_t i nodi non rilevanti
        else:
            X_t.append(X_train[j, :])
            Y_t.append(Y_train[j])
    # Infine faccio l'unione X_t con X_val ed anche  Y_t con Y_val cioè unisco i nodi non rilevanti ai già presenti nodi in X_val e Y_val
    for j in range(len(Y_val)):
        X_t.append(X_val[j])
        Y_t.append(Y_val[j])

    # restituisco tutti i numpy array aggiornati
    return np.asarray(X_temp), np.asarray(Y_temp), np.asarray(X_t), np.asarray(Y_t)