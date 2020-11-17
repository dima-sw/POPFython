import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s

from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp

if __name__ == '__main__':
    # Loading a .txt file to a numpy array
    txt = l.load_txt(r'C:\Users\TheDimitri\Documents\GitHub\POPFython\data\nug.txt')

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Splitting data into training and validation sets
    X_train, X_val, Y_train, Y_val = s.split(
        X, Y, percentage=0.5, random_state=1)

    # Creates a SupervisedOPF instance
    opf = SSupervisedPOPF(processi=mp.cpu_count(),distance='log_squared_euclidean',
                        pre_computed_distance=None)

    # Performs the pruning procedure
    opf.prune(X_train, Y_train, X_val, Y_val,6,0.05, n_iterations=10)
