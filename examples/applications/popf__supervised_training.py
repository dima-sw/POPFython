import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp
import time
from examples.Testing_stuff.createData import creaGig

if __name__ == '__main__':

                i=0.1
                file = open("../../data/tryConda.txt", "w")
                # Loading a .txt file to a numpy array
                txt = l.load_txt(r'C:\Users\TheDimitri\Documents\GitHub\POPFython\data\miniboo.txt')
                tagli = 10
                # Parsing a pre-loaded numpy array
                X, Y = p.parse_loader(txt)
                # Creates a SupervisedPOPF instance
                opf = SSupervisedPOPF(processi=6, tagli=tagli
                                      , distance='squared_euclidean',
                                      pre_computed_distance=None)

                while i <= 0.5:
                    # Splitting data into training and testing sets
                    X_train, X_test, Y_train, Y_test = s.split(
                        X, Y, percentage=i,percentage2=i, random_state=1)

                    t1 = time.time()
                    opf.numba_Fit(X_train, Y_train)
                    L2,P2=opf.numba_Predict(X_test)

                    acc = g.opf_accuracy(Y_test, L2)

                    t2 = time.time()
                    tot = t2 - t1

                    print("Tempo: ", tot)
                    print(f'Accuracy: {acc}')
                    file.write(str(len(X_train)*2)+" "+str(tot)+"\n")
                    i=i+0.1
                file.close()