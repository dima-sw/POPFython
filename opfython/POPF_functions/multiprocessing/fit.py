import numpy as np
import opfython.utils.constants as c
import opfython.utils.logging as log
from opfython.core import Subgraph
from multiprocessing import JoinableQueue, Queue, Array
from opfython.POPF_functions.POPF_functions import creaProcFit, creaTagli, calcMin
import time

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
    opf.mult_Find_Prototypes()

    start = time.time()

    """Questi 3 array li useremo durante la concorrenza
                P-> array dei predecessori
                C-> array dei costi
                U-> array degli used per vedere se gia' abbiamo usato il nodo
                L-> array delle label
    """
    P = Array('i', opf.subgraph.n_nodes, lock=False)
    C = Array('f', opf.subgraph.n_nodes, lock=False)
    L = Array('i', opf.subgraph.n_nodes, lock=False)
    U = Array('i', opf.subgraph.n_nodes, lock=False)

    # Inizializzo gli array e prendo il primo prototipo
    primo = initGraphFit(opf, U, C, P, L)

    # Mi conservo i processi, serve durante il pruring per non intasare la memoria con processi
    processi = []

    work = JoinableQueue()
    result = Queue()

    # creo e faccio partire i processi
    creaProcFit(train, processi, opf._processi, opf, P, C, L, U, work, result)

    """parti= [[0,n_nodi/tagli],...,[(tagli-1)*(n_nodi/tagli),n_nodi]]"""  # partiziono in n parti uguali con n=tagli
    parti = []
    creaTagli(opf._tagli, parti, opf.subgraph.n_nodes)

    # prendo il primo prototipo
    s = primo

    # Inizia il vero e proprio training
    fitCompute(opf, s, U, C, work, result, parti)

    # termino i processi (Utile per non intasare la memoria con processi quando si usa il pruning)
    for i in range(opf._processi):
        processi[i].terminate()

    # aggiorno pred e label dei nodi
    for j in range(0, opf.subgraph.n_nodes):
        opf.subgraph.nodes[j].pred = P[j]
        opf.subgraph.nodes[j].predicted_label = L[j]

    # The subgraph has been properly trained
    opf.subgraph.trained = True

    # Ending timer
    end = time.time()

    # Calculating training task time
    train_time = end - start

    logger.info('Classifier has been fitted.')
    logger.info('Training time: %s seconds.', train_time)


# Inizializzo gli array per il training in base se un nodo è un prototipo o meno
def initGraphFit(opf, U, C, P, L):
    flag = True  # per prendere il primo prototipo

    # For each possible node
    for i in range(opf.subgraph.n_nodes):
        U[i] = 0
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


# La parte del training
def fitCompute(opf, s, U, C, work, result, parti):
    """Percentuali per la stampa del tempo stimato e a che % stiamo"""
    """percent = 0
    percOld = 1
    flagTime = True """
    # quando avrò computato tutti i nodi s sarà = -1
    while s != -1:
        """tempo stimato e %"""
        """if flagTime:
            startPerc = time.time()
            flagTime = False"""

        # marchio il nodo come usato
        U[s] = 1

        # Lo aggiungo alla lista ordinata
        opf.subgraph.idx_nodes.append(s)

        # Gathers its cost
        opf.subgraph.nodes[s].cost = C[s]

        # Metto nella JoinableQueue tutte le partizioni e il nodo su cui operare
        for i in range(opf._tagli):
            work.put((parti[i][0], parti[i][1], s))

        # Aspetto che i worker finiscano
        work.join()

        # prendo il più piccolo s
        s = calcMin(result)


    # Tengo traccia a che punto sta, per file grossi è comodo
    """percnew = (percent / opf.subgraph.n_nodes) * 100
        #Sempre per tenere traccia
        if (percnew > percOld):
            endPerc = time.time()
            percOld += 1
            print("Training %" + str(int(percnew)))

            ("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
            flagTime = True
        percent += 1"""


# Parte concorrente del training
def train(opf, P, C, L, U, work, result):

    while True:

        # vedo se c'è un range sul quale lavorare
        r1, r2, s = work.get()

        # s1 dovrà essere il nodo non usato con il costo più piccolo,
        # workInRange è proprio il lavoro che svolge il processo nel range r1,r2 (for interno)
        s1 = workInRange(opf, s, r1, r2, C, L, P, U)

        # s1=None significa che ogni nodo di questo range è già stato used e restituiamo -1
        if s1 == None:
            result.put((-1, -1))
        # restituisco il risultato al processo principale
        else:
            result.put((s1, C[s1]))
        # finisco su questo range
        work.task_done()



def workInRange(opf, s, r1, r2, C, L, P, U):
    s1 = None
    #print("Numero processo: ", i)
    # lavoro solo nel range preso dalla work.get() r1,r2
    for t in range(r1, r2):

        # se non stiamo confrontando lo stesso nodo con se stesso
        if s != t:

            # se il costo di s è più piccolo di t
            if C[t] > C[s]:

                # calcolo la distanza tra s e t
                weight = opf.calcWeight(s, t)

                # Il costo corrente sarà il massimo tra il costo dell'arco tra i due nodi (weight, l'arco in realtà non esiste) e il nodo s
                current_cost = np.maximum(C[s], weight)

                # If current cost is smaller than `t` node's cost
                if current_cost < C[t]:
                    # aggiorno la label di t che sarà uguale a quella di s
                    L[t] = L[s]
                    # aggiorno il costo di t con quello corrente
                    C[t] = current_cost
                    P[t] = s

            # se s1 non è stato ancora assegnato oppure il costo di s1>costo t1
            if (s1 == None or C[s1] > C[t]):
                # e se il nodo t non è stato già used aggiorno s1
                if U[t] == 0:
                    s1 = t
    return s1

