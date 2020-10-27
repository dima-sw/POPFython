import random

def creaGig():
    giganti=open("../../data/nug.txt", "w")
    #altezza, età, peso,
    carU=[(140,200),(0,80),(30,100)]
    carG = [(200, 500), (0, 200), (80, 500)]
    carN = [(5, 140), (0, 20), (1, 40)]

    tutto=[carN,carU,carG]


<<<<<<< HEAD:examples/Testing stuff/createData.py
    for j in range(5000):
=======
    for j in range(2000):
>>>>>>> 7b3b242eb904dd7f22262c8dc54b8708b8089cfd:examples/createData.py
        i= random.randint(0,2)

        giganti.write(str(j)+" "+str(i+1)+" "+str(random.uniform(tutto[i][0][0],tutto[i][0][1]))+" "+str(random.uniform(tutto[i][1][0],tutto[i][1][1]))+" "+str(random.uniform(tutto[i][2][0],tutto[i][2][1]))+'\n')

    giganti.close()

creaGig()

