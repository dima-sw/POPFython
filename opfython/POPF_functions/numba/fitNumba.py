import time
import numpy as np
import opfython.utils.constants as c
import opfython.utils.logging as log
from opfython.core import Subgraph
from numba import njit,prange, set_num_threads
from opfython.POPF_functions.POPF_functions import  creaTagliNP, calcMin


logger = log.get_logger(__name__)

#Questa funzione calcola il minimo s1 in termine di costo
def calcMin(C,res):
    l=res[0]
    for i in range(1,len(res)):
        if l==-1 or (res[i]!=-1 and C[res[i]]<C[l]):
            l=res[i]
    return l

#Inizializzo gli array in base se sono prototipi o meno e restituisco il primo prototipo
def initGraphFit(opf,U,C,P,L):
    flag = True  # per prendere il primo prototipo

    # For each possible node
    for i in range(opf.subgraph.n_nodes):
        # Checks if node is a prototype
        if opf.subgraph.nodes[i].status == c.PROTOTYPE:

            """Se e' un prototipo Costo=0, Pred=nil, Label=la stessa del nodo"""
            C[i] = 0
            P[i] = c.NIL
            L[i] = opf.subgraph.nodes[i].label
            # prendo il primo prototipo
            if flag:
                primo = i
                flag = False

        # If node is not a prototype
        else:
            """se non è un prototipo Costo=MAX, Pred=lo stesso del nodo,  label=nil"""

            C[i] = c.FLOAT_MAX
            P[i] = opf.subgraph.nodes[i].pred
            L[i] = c.NIL

    return primo


def fit(opf, X_train, Y_train, I_train=None):
    """Fits data in the classifier.

    Args:
        X_train (np.array): Array of training features.
        Y_train (np.array): Array of training labels.
        I_train (np.array): Array of training indexes.

    """

    logger.info('Fitting classifier ...')





    # Creating a subgraph
    opf.subgraph = Subgraph(X_train, Y_train, I=I_train)

    # Finding prototypes
    opf.numba_Find_Prototypes(X_train)

    # Set il numero di thread da utilizzare
    set_num_threads(opf._processi)

    start = time.time()

    """Questi 3 array li useremo durante la concorrenza
                    P-> array dei predecessori
                    C-> array dei costi
                    U-> array degli used per vedere se gia' abbiamo usato il nodo
                    L-> array delle label
        """
    P = np.full(opf.subgraph.n_nodes, 0, dtype=np.int)
    C = np.full(opf.subgraph.n_nodes, 0, dtype=np.float64)
    U = np.full(opf.subgraph.n_nodes, 0, dtype=np.int)
    L = np.full(opf.subgraph.n_nodes, 0, dtype=np.int)

    #inizializzo il grafo e prendo il primo prototipo
    primo = initGraphFit(opf, U, C, P, L)

    #Creo le slice
    slices=creaTagliNP(opf._processi,opf.subgraph.n_nodes)
    #Creo l'array dei risultati
    res = np.full(opf._processi, -1, dtype=np.int)

    #Inizia il training
    fitCompute(opf,primo,P,L,U,C,res,slices,X_train)

    # aggiorno pred e label dei nodi
    for j in range(0, opf.subgraph.n_nodes):
        opf.subgraph.nodes[j].pred = (int) (P[j])
        opf.subgraph.nodes[j].predicted_label = (int)(L[j])
        opf.subgraph.nodes[j].cost = (C[j])

    # The subgraph has been properly trained
    opf.subgraph.trained = True

    # Ending timer
    end = time.time()

    # Calculating training task time
    train_time = end - start

    logger.info('Classifier has been fitted.')
    logger.info('Training time: %s seconds.', train_time)


def fitCompute(opf,s,P,L,U,C,res,slices,xtrain):

    while s!=-1:
        U[s]=1
        #Appendo il nodo a una lista ordinata
        opf.subgraph.idx_nodes.append(s)
        #Faccio partire il work sul nodo s
        worker(P,C,U,L,s,xtrain,slices,res,opf._processi)

        #Il prossimo s sara' il nodo col costo più piccolo
        s=calcMin(C,res)



@njit(fastmath=True,nogil=True,parallel=True,cache=True)
def worker(P,C,U,L,s,xtrain,slice,res,n_threads):

    #Suddivido il lavoro in N threads ed a ciascuno do una slice sulla quale lavorare
    for i in prange(n_threads):
        work(P,C,U,L,s,xtrain,res,slice[i][0],slice[i][1],i)



@njit(fastmath=True,nogil=True,cache=True)
def work(P,C,U,L,s,xtrain,res,r1,r2,i):
    s1 = -1
    #Lavoro solo su una slice da r1 ad r2
    for t in range(r1, r2):
            if s != t:
                if C[t]>C[s]:
                    #Calcolo la distanza
                    weight = calcWeight(xtrain[s], xtrain[t])
                    #Il costo corrente sara' il massimo tra il costo di s e la distanza tra s e t
                    current_cost=np.maximum(C[s],weight)
                    if current_cost < C[t]:
                        # aggiorno la label di t che sarà uguale a quella di s
                        L[t] = L[s]
                        # aggiorno il costo di t con quello corrente
                        C[t] = current_cost
                        #Aggiorno il predecessore di t
                        P[t] = s
                #Vedo se t possa diventare s1
                if (s1 == -1 or C[s1] > C[t]):
                    if U[t] == 0:
                     s1 = t
    #Aggiorno il risultato
    res[i] = s1

#Calcolo della distanza euclidiana
@njit(fastmath=True,nogil=True,cache=True)
def calcWeight(x,y):
    dist = 0
    for i in range(len(x)):
        dist += (x[i] - y[i]) * (x[i] - y[i])

    return dist
