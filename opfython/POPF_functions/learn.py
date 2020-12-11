import numpy as np
import copy
import opfython.utils.constants as c
import opfython.utils.logging as log


logger = log.get_logger(__name__)


def learn(opf, xt, yt, xv, yv,n_iterations=10,variazione=0.001):
    """
    Args:
        opf: il nostro classificatore
        xt: numpyArray dei campioni per il training
        yt: numpyArray delle label per il training
        xv: numpyArray dei campioni per la convalida
        yv: numpyArray delle label per la convalida
        n_iterations: quante iterazioni del learning vogliamo fare
        variazione: se nell'iterazione attuale abbiamo un cambiamento dell'accuratezza (acc attuale- acc precedente) più piccolo di variazione
                                                                                        possiamo finere il learning
    """

    # Devo salvare i set di training e i set di classification del miglior classificatore
    X_val = copy.deepcopy(xv)
    X_train = copy.deepcopy(xt)
    Y_val = copy.deepcopy(yv)
    Y_train = copy.deepcopy(yt)

    max_acc=-1
    prev_acc=0




    for it in range(0,n_iterations):

        # Calcolo il numero di label diverse es. [1,2,3,2,2,2] -> 3
        num_class = np.max(Y_val)

        #Calcolo il numero totale di ogni label es. [1,3,2,2,2,2] -> [1,3,1]
        _, count_label = np.unique(Y_val, return_counts=True)

        #Lista dei campioni misclassificati
        LM=[]

        #Training
        opf.fit(X_train, Y_train)

        #Numpy Array dei Falsi positivi (FP) e falsi negativi (FN) a ogni iterazione li azzero
        FP=np.zeros(num_class)
        FN=np.zeros(num_class)

        #Classificazione
        L2,P2=opf.pred(X_val)

        #Aggiorno FP ed FN e calcolo l'accuratezza
        acc=cur_acc(L2,Y_val,FP,FN,LM,count_label,num_class)

        #Se l'accuratezza e' migliore di quella massima, mi salvo il classificatore
        if acc>max_acc:
            max_acc=acc
            best_istance=copy.deepcopy(opf.subgraph)

            #Aggiorno anche le liste cosi da avere i set giusti (Indispensabile per il pruning)
            xt[:] = X_train[:]
            xv[:] = X_val[:]
            yt[:] = Y_train[:]
            yv[:] = Y_val[:]

        #Rimpiazzo i campioni misclassificati di X_val e Y_val con campioni non prototipi di X_train e Y_train (randomicamente)
        swap_err(X_train,Y_train,X_val,Y_val,opf.subgraph,LM)

        var=abs(acc-prev_acc)
        prev_acc=acc

        logger.info('Current accuracy: %s | Variation: %s | Max_accuracy: %s', acc, var, max_acc)

        #se la variazione e' minima nell'accuratezza possiamo terminare il learning
        if(var<=variazione):
            break

    #Mi prendo il classificatore migliore
    opf.subgraph=best_istance
    return max_acc


def cur_acc(L2,Y_val,FP,FN,LM,count_label,num_class):


    for t in range(0, len(Y_val)):
        # se t e' misclassificato incremento i falsi positivi e falsi negativi, aggiungo t ad LM
        if L2[t] != Y_val[t]:
            FP[L2[t] - 1] += 1
            FN[Y_val[t] - 1] += 1

            LM.append(t)

    #Calcolo l'accuratezza secondo l'articolo 'Supervised Pattern Classification based on Optimum-Path Forest'
    Err_FN = FN[:] / count_label
    Err_FP = FP[:] / (np.sum(count_label) - count_label)

    Error = np.sum(Err_FN + Err_FP)

    acc = 1 - (Error) / (2 * num_class)

    return acc


def swap_err(X_train,Y_train,X_val,Y_val,subgraph,LM,):
    #Mi prendo quanti non prototipi ci sono nel classificatore
    non_prot = len(X_train) - subgraph.n_prot

    for t in LM:

        #Non prototipi attuali serve perche' ci potrebbero essere più t in LM che nodi non prototipi
        curr_non_prot = non_prot
        while curr_non_prot > 0:

            #Genero un numero casuale UNIFORME
            ran = (int)(np.random.uniform(0, len(X_train), 1))

            #Se non è un prototipo allora faccio lo swap
            if (subgraph.nodes[ran].status != c.PROTOTYPE):
                X_train[ran, :], X_val[t, :] = X_val[t, :], X_train[ran, :]
                Y_train[ran], Y_val[t] = Y_val[t], Y_train[ran]
                non_prot -= 1

                break
            else:
                curr_non_prot -= 1


