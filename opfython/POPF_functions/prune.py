import numpy as np
import opfython.utils.constants as c
import opfython.utils.logging as log
import opfython.math.general as g


logger=log.get_logger(__name__)

def prune(opf, X_train, Y_train, X_val, Y_val,M_loss,fit,pred, n_iterations=10):
    """
    Args:
        opf: il nostro classificatore
        X_train: numpyArray dei campioni per il training
        Y_train: numpyArray delle label per il training
        X_val: numpyArray dei campioni per la convalida
        Y_val: numpyArray delle label per la convalida
        M_loss: (float) di quanto può scendere l'accuratezza
        n_iterations: (int) numero massimo di iterazioni del learning
    """

    logger.info('Pruning classifier ...')

    #Primo learning
    opf.learn(X_train,Y_train,X_val,Y_val,tfit=fit, tpred=pred,n_iterations=n_iterations)

    """L2=label P2=predecessore, output dal classificatore"""
    L2, P2= pred(X_val)
    acc=g.accuracy(Y_val,L2)

    # Prendo il numero di nodi iniziali del grafo
    initial_nodes = opf.subgraph.n_nodes

    # Faccio partire il pruring
    pruringRun(opf, acc, M_loss, n_iterations, X_train, Y_train, X_val, Y_val,P2,fit,pred)

    # Prendo i nodi a fine pruring
    final_nodes = opf.subgraph.n_nodes
    logger.info('Initial number of nodes: %s , final number of nodes: %s,', initial_nodes, final_nodes)
    # Calculating pruning ratio
    prune_ratio = 1 - final_nodes / initial_nodes

    logger.info('Prune ratio: %s.', prune_ratio)



def pruringRun(opf, acc, M_loss,n_iterations, X_train, Y_train, X_val, Y_val,P2,fit,pred):
    # tmp= accuratezza attuale
    tmp = acc
    # flag serve per tenere traccia se ci sia ancora almeno un nodo non rilevante
    flag = True

    # mentre l'accuratezza attuale è >= (dell'accuratezza massima iniziale - M_loss) e mentre ci sta almeno un nodo non Rilevante (flag)
    while abs(acc - tmp) <= M_loss and flag:

        #Suppongo che tutti i nodi siano rilevanti
        flag = False
        # Lunghezza prima di rimuovere i nodi irrilevanti
        lunIn = len(X_train)


        # Rimuovo i nodi irrillevanti
        # I nodi irrellevanti li rimuovi da X_train e Y_train e li aggiungo a X_val e Y_val
        X_train, Y_train,X_val,Y_val = pruringUpdateList(opf.subgraph,X_train, Y_train, X_val, Y_val,P2)

        # Lunghezza dopo aver tolto i nodi irrilevanti
        lunFin = len(X_train)

        # Se la lunghezza non cambia significa che possiamo terminare il pruring, altrimenti facciamo partire un nuovo learning
        if lunFin < lunIn:

            #la lunghezza e' diversa significa che abbiamo spostato almeno un nodo da X_trai e Y_train a X_val e Y_val
            flag = True

            # Faccio partire un nuovo learning
            opf.learn(X_train, Y_train, X_val, Y_val,tfit=fit, tpred=pred, n_iterations=n_iterations)
            """L2=label P2=predecessore, output dal classificatore"""
            L2,P2 = pred(X_val)
            # Calcolo l'accuratezza
            tmp = g.accuracy(Y_val, L2)
            logger.info('Current accuracy: %s.', tmp)


def pruringUpdateList(subgraph,X_train, Y_train, X_val, Y_val,P2):
    tmp_x=[]
    tmp_y=[]


    #Marchio i nodi rilevanti seguendo optimum path usato per la classificazione
    while not P2.empty():
        s=P2.get()
        while s!=c.NIL:
            subgraph.nodes[s].relevant = c.RELEVANT
            s=subgraph.nodes[s].pred


    #i nodi rilevanti devono far parte ancora di X_train e Y_train per il prossimo learning
    #quelli non rilevanti invece li aggiungo a X_val e Y_val
    for j, n in enumerate(subgraph.nodes):
        if n.relevant:
            tmp_x.append(X_train[j,:])
            tmp_y.append(Y_train[j])
        else:
            X_val = np.vstack([X_val,X_train[j,:]])
            Y_val = np.insert(Y_val,len(Y_val), Y_train[j])

    return np.asarray(tmp_x), np.asarray(tmp_y), X_val,Y_val