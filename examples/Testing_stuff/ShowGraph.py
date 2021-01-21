import matplotlib.pyplot as plt

"""These are some benchmarks that I did on my PC using different implementaton

@Author: Dmytro Lysyhanych
"""


def ShowGraph(*args):

    l=len(args)
    lab=(int)(l/2)
    for i in range((int)(l/2)):
        x1=[]
        y1=[]
        with open(args[i]) as f:
            for line in f:
                iii=int(line.split(" ")[0])
                if(iii<=400000 and iii>=0):
                    x1.append(int(line.split(" ")[0]))
                    y1.append(float(line.split(" ")[1]))
                if 'str' in line:
                    break
        plt.plot(x1, y1, label=args[lab][0], color=args[lab][1])
        lab+=1

    plt.title('Dataset:People\n Features per campione:3')
    plt.xlabel("Numero campioni")
    plt.ylabel("Tempo (s)")
    plt.legend()
    plt.show()

c1="tab:blue"
c2="tab:orange"
c3="tab:green"
c4="red"
c5="tab:purple"
c6="tab:brown"
c7="tab:pink"
c8="tab:gray"
c9="tab:olive"
c10="tab:cyan"
c11="dimgray"
c12="darkslateblue"
c13="purple"
c14="darkred"

f1=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\graficoNUMBA6.txt"
l1="POPF-6 (numba)"
f2=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\graficoNUMBA4.txt"
l2="POPF-4 (numba)"
f3=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\graficoNUMBA2.txt"
l3="POPF-2 (numba)"
f4=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\graficoNUMBA4.txt"
l4="POPF-4 (numba)"
f5=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\cyt.txt"
l5="seq. OPF (Cython)"
f6=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\Cython2.txt"
l6="POPF-2 (Cython)"
f7=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\Cython4.txt"
l7="POPF-4 (Cython)"
f8=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\Cython6.txt"
l8="POPF-6 (Cython)"
f9=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\seqd.txt"
l9="seq. OPF (Python+Cython distance)"
f10=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\mpd.txt"
l10="POPF-6 (Python multiprocessing+Cython distance)"
f11=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\grafico.txt"
l11="POPF-6 (Python multiprocessing)"
f12=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap\graficoNonConc.txt"
l12="seq. OPF (Python)"


f01=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50graficoNUMBA6.txt"
l01="POPF-6 (numba)"
f02=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50graficoNUMBA4.txt"
l02="POPF-4 (numba)"
f03=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50graficoNUMBA2.txt"
l03="POPF-2 (numba)"
f04=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50graficoNUMBA4.txt"
l04="POPF-4 (numba)"
f05=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50Cython.txt"
l05="seq. OPF (Cython)"
f06=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50Cython2.txt"
l06="POPF-2 (Cython)"
f07=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50Cython4.txt"
l07="POPF-4 (Cython)"
f08=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50Cython6.txt"
l08="POPF-6 (Cython)"
f09=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50eopfDist.txt"
l09="seq. OPF (Python+Cython distance)"
f010=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50popfDist.txt"
l010="POPF-6 (Python multiprocessing+Cython distance)"
f011=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50popf.txt"
l011="POPF-6 (Python multiprocessing)"
f012=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50eeeoopf.txt"
l012="seq. OPF (Python)"
f013=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\50popf6C.txt"
l013="POPF-6 (C)"
f014=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grap2\seqC.txt"
l014="seq. OPF (C)"




#ShowGraph(file1,file2,(label1,color1),(label2,color2))

ShowGraph(f03,f02,f01,f06,f07,f08,(l03,c3),(l02,c2),(l1,c1),(l6,c6),(l7,c7),(l8,c8))