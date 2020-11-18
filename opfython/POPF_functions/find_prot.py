from multiprocessing import JoinableQueue, Queue,Array
import opfython.utils.constants as c
import opfython.utils.logging as log
from opfython.POPF_functions.POPF_functions import creaProcFit,creaTagli,calcMin
import time

logger = log.get_logger(__name__)

def _find_prototypes(self, tagli):
    """Find prototype nodes using the Minimum Spanning Tree (MST) approach.
    """

    logger.debug('Finding prototypes ...')

    start = time.time()

    """Questi 3 array li useremo durante la concorrenza
        P-> array dei pred
        C-> array dei costi
        U-> array degli used per vedere se gia' abbiamo usato il nodo
    """
    P = Array('i', self.subgraph.n_nodes, lock=False)
    C = Array('f', self.subgraph.n_nodes, lock=False)
    U = Array('i', self.subgraph.n_nodes, lock=False)

    # Inizialmente C =max e P= nil
    for i in range(self.subgraph.n_nodes):
        C[i] = c.FLOAT_MAX
        P[i] = c.NIL

    # Marking first node without any predecessor
    self.subgraph.nodes[0].pred = c.NIL
    P[0] = c.NIL

    # primo nodo
    p = 0

    # Creating a list of prototype nodes
    prototypes = []
    processi = []

    work = JoinableQueue()
    result = Queue()

    # creo i processi
    creaProcFit(protParal, processi, self._processi,self, P, C, U, work, result)
    parti = []
    creaTagli(tagli, parti, self.subgraph.n_nodes)

    # Inizia ufficialmente a trovare i prototipi con MST

    start_find_prototypes(self, p, U, P, C, prototypes, tagli, work, parti, result)

    # termino i processi (Utile per non intasare la memoria con processi quando si usa il pruring)
    for i in range(self._processi):
        processi[i].terminate()

    # Aggiorno il grafo
    for i in range(self.subgraph.n_nodes):
        self.subgraph.nodes[i].pred = P[i]

    end = time.time()
    fittime = end - start
    logger.debug('Prototypes: %s.', prototypes)
    logger.info('Prototypes found in: %s seconds.', fittime)


def updateProt(self, prototypes, p, pred):
    # Checks if the label of current node is the same as its predecessor
    if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
        # If current node is not a prototype
        if self.subgraph.nodes[p].status != c.PROTOTYPE:
            # Marks it as a prototype
            self.subgraph.nodes[p].status = c.PROTOTYPE
            # Appends current node identifier to the prototype's list
            prototypes.append(p)

        # If predecessor node is not a prototype
        if self.subgraph.nodes[pred].status != c.PROTOTYPE:
            # Marks it as a protoype
            self.subgraph.nodes[pred].status = c.PROTOTYPE

            # Appends predecessor node identifier to the prototype's list
            prototypes.append(pred)


def start_find_prototypes(self, p, U, P, C, prototypes, tagli, work, parti, result):
    """percent = 0
                   percOld = 1
                   flagTime = True """

    # Finchè tutti i nodi non sono used
    while p != -1:

        """if flagTime:
            startPerc = time.time()
            flagTime = False"""

        # Gathers its cost
        self.subgraph.nodes[p].cost = C[p]

        # And also its predecessor
        pred = P[p]

        # Nodo p=used
        U[p] = 1

        # If the predecessor is not NIL update prototypes
        if pred != c.NIL:
            updateProt(self,prototypes, p, pred)

        # Metto le suddivisioni e p in work
        for i in range(tagli):
            work.put((parti[i][0], parti[i][1], p))

        # Aspetto che finiscono i processi
        work.join()

        # prendo il minimo
        p = calcMin(result)

        """percnew = (percent / self.subgraph.n_nodes) * 100
        if (percnew > percOld):
            endPerc = time.time()
            percOld += 1
            print("Prototypes %" + str(int(percnew)))
            print("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
            flagTime = True
        percent += 1"""





# La parte concorrente di find_prototypes
def protParal(self, P, C, U, work, result):
    while True:
        # prendo un range e il nodo p
        r1, r2, p = work.get()

        s1 = None

        # lavoro su un range
        for q in range(r1, r2):
            # Vedo se il nodo non è ancora stato usato
            if U[q] == 0:
                # If `p` and `q` identifiers are different
                if p != q:

                    weight = self.calcWeight(p, q)
                    # If current arc's cost is smaller than the path's cost
                    if weight < C[q]:
                        # Marks `q` predecessor node as `p`
                        P[q] = p

                        # Aggiorno il costo
                        C[q] = weight
                    if (s1 == None or C[s1] > C[q]):
                        # e se il nodo non è stato già used
                        if U[q] == 0:
                            s1 = q

        # s1=None significa che ogni nodo di questo range è stato used
        if s1 == None:
            s1 = -1
        # restituisco il risultato al processo principale il più piccolo s1 e il suo costo
        result.put((s1, C[s1]))
        work.task_done()