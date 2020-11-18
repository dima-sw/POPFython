import numpy as np
import opfython.utils.constants as c
import opfython.utils.logging as log

logger = log.get_logger(__name__)


def prune(opf, X_train, Y_train, X_val, Y_val, tagli, M_loss, n_iterations=10):
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
    acc = opf.learn(X_train, Y_train, X_val, Y_val, tagli, n_iterations=n_iterations)

    # Prendo il numero di nodi iniziali del grafo
    initial_nodes = opf.subgraph.n_nodes

    # Faccio partire il pruring
    pruringRun(opf,acc, M_loss, tagli, n_iterations, X_train, Y_train, X_val, Y_val)

    # Prendo i nodi a fine pruring
    final_nodes = opf.subgraph.n_nodes
    logger.info('Initial number of nodes: %s , final number of nodes: %s,', initial_nodes, final_nodes)
    # Calculating pruning ratio
    prune_ratio = 1 - final_nodes / initial_nodes

    logger.info('Prune ratio: %s.', prune_ratio)


def pruringRun(opf, acc, M_loss, tagli, n_iterations, X_train, Y_train, X_val, Y_val):
    # tmp= accuratezza attuale
    tmp = acc
    # flag serve per tenere traccia se ci sia ancora almeno un nodo non rilevante
    flag = True

    # mentre l'accuratezza attuale è >= (dell'accuratezza massima iniziale - M_loss) e mentre ci sta almeno un nodo non Rilevante (flag)
    while abs(acc - tmp) <= M_loss and flag:
        # Rimuovo i nodi irrillevanti, aggiorno X_train, Y_train, X_val,Y_val e vedo se ci sta almeno un nodo non rilevante con flag
            #I nodi irrellevanti li rimuovi da X_train e Y_train e li aggiungo a X_val e Y_val
        flag, X_train, Y_train, X_val, Y_val = pruringUpdateList(opf.subgraph.nodes, X_train, Y_train, X_val, Y_val)

        # Faccio il learning e prendo l'accuratezza
        tmp = opf.learn(X_train, Y_train, X_val, Y_val, tagli, n_iterations=n_iterations)
        logger.info('Current accuracy: %s.', tmp)


def pruringUpdateList(nodes, X_train, Y_train, X_val, Y_val):
    # Liste temporanee per X_train, Y_trrain, X_val e Y_val
    X_temp, Y_temp = [], []
    X_t, Y_t = [], []

    # Se flag rimanse false significa che tutti i nodi sono rilevanti e quindi possiamo terminare il pruning
    flag = False
    for j, n in enumerate(nodes):
        # Aggiungo alle liste temporanee X_temp, Y_temp i nodi rilevanti
        if n.relevant != c.IRRELEVANT:
            X_temp.append(X_train[j, :])
            Y_temp.append(Y_train[j])
        # Aggiungo alle liste temporanee X_t e Y_t i nodi non rilevanti
        else:
            # flag=true non tutti i nodi sono rilevanti
            flag = True
            X_t.append(X_train[j, :])
            Y_t.append(Y_train[j])
    # Infine faccio l'unione X_t U X_val, Y_t U Y_val cioè unisco i nodi non rilevanti ai già presenti nodi in X_val e Y_val
    for j in range(len(Y_val)):
        X_t.append(X_val[j])
        Y_t.append(Y_val[j])

    # restituisco flag e tutti i numpy array
    return flag, np.asarray(X_temp), np.asarray(Y_temp), np.asarray(X_t), np.asarray(Y_t)