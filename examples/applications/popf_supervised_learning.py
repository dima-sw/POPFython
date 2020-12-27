import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp
import time

if __name__ == '__main__':

        tagli = 10
        # Loading a .txt file to a numpy array
        txt = l.load_txt(r'C:\Users\TheDimitri\Documents\GitHub\POPFython\data\miniboo.txt')

        # Parsing a pre-loaded numpy array
        X, Y = p.parse_loader(txt)

        # Splitting data into training and testing sets
        X_train, X_test, Y_train, Y_test = s.split(
            X, Y, percentage=0.2,percentage2=0.4, random_state=1)



        print(X_train)

        # Creates a SupervisedPOPF instance
        opf = SSupervisedPOPF(processi=mp.cpu_count(),tagli=tagli
                                 ,distance='squared_euclidean',
                                pre_computed_distance=None)
        #calcolo il tempo
        t1 = time.time()
        opf.learn(X_train, Y_train,X_test,Y_test,n_iterations=7)


        t2 = time.time()
        tot = t2 - t1

        print("Tempo learning: ", tot)


