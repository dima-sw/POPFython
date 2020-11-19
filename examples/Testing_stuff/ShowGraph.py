import matplotlib.pyplot as plt

def ShowGraph(f1,f2,l1,l2):
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    with open(f1) as f:
        for line in f:
            x1.append(int(line.split(" ")[0]))
            y1.append(float(line.split(" ")[1]))
            if 'str' in line:
                break
    with open(f2) as f:
        for line in f:
            x2.append(int(line.split(" ")[0]))
            y2.append(float(line.split(" ")[1]))
            if 'str' in line:
                break

    plt.plot(x1, y1, label="Concorrente")
    plt.plot(x2, y2, label="Non Concorrente")
    plt.title('Numero features per ogni campione=3!')
    plt.xlabel(l1)
    plt.ylabel(l2)
    plt.legend()
    plt.show()


f1=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\grafico.txt"
f2=r"C:\Users\TheDimitri\Documents\GitHub\POPFython\data\graficoNonConc.txt"
l1="Numero campioni"
l2="Tempo (s)"

ShowGraph(f1,f2,l1,l2)