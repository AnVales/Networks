# Implementation of Erdos-Renyi Model 
import numpy as np
from numpy import random 
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt

# VARIABLES
N=150 # nodos
p=0.3 # probabilidad de link

# RECORDATORIO DE FORMULAS
# average_degree = n*p
# average_number_links = p*n*(n-1)/2

# PREPARACIÓN DE MATRIZ DE ADYACENCIA: Matriz llena de 0s
size = (N, N)
matriz0 = np.zeros(size)

# FUNCIÓN PONER UNOS EN LA MATRIZ
def random_net(N, p, matriz):
    for i in range(N):
        for j in range(N-1):
            r = random.random_sample(1)
            if r<p:
                matriz[i,j]=1
                matriz[j,i]=1
    return matriz

# OBTENER LA MATRIZ DE ADYACENCIA
matriz_random = random_net (N, p, matriz0)
# print(matriz_random)

# FUNCIÓN HACER DICCIONARIO DE LA MATRIZ
def matrizADic(matriz):
    dic = {}

    for r in range(len(matriz)):
        for c in range(len(matriz[r])):
            if matriz[r][c] == 1: # Si en un valor encuentra un 1
                tuplaAux = (r, c) # Guardamos la fila y la columna de donde está 
                if r not in dic: # Si no tenemos el diccionario se crea
                    dic[r] = {}
                dic[r][c] = tuplaAux # Metemos en el diccionario, como valor la relación

                if c not in dic: # Se crea un diccionario del gen
                    dic[c] = {}
                
    return dic

# OBTENER EL DICCIONARIO
hash_matriz = matrizADic(matriz_random)
# print(hash_matriz)

# VAMOS A USAR networkx
adj = np.asmatrix(matriz_random)
# print(adj)

# CREAMOS EL OBJETO networkx
G = nx.from_numpy_matrix(adj)

# VEMOS INFORMACIÓN
print("Nº of edges", G.number_of_edges())
print("Nº of nodes", G.number_of_nodes())
print("Diameter:", nx.algorithms.distance_measures.diameter(G))
print("average degree connectivity of graph:", nx.algorithms.assortativity.average_degree_connectivity(G))
print("Clustering:", nx.algorithms.approximation.clustering_coefficient.average_clustering(G))
print("Number of connected components:", nx.number_connected_components(G))
print("Is regular?", nx.algorithms.regular.is_regular(G))
nx.draw(G, node_color="r", node_size=150, with_labels=True)
plt.show()
diameter=nx.algorithms.distance_measures.diameter(G)

# RECORRAMOS LA RED
def rec_search(key, count, net_hash, sum_hash, diam, timerumor_matrix, rumor):
    count = count + 1
    p=0.5
    values = net_hash[key]
    next_key_list = random.choice(list(values))

    # metemos el rumor
    if sum_hash[next_key_list]==0 and count == 1:
        sum_hash[next_key_list]=1
        rumor=rumor+1

    # contamos el rumor, si se cuenta o no se cuenta 
    if sum_hash[next_key_list]==0 and count != 1:
        r = random.random_sample(1)
        if p>r:
            print(next_key_list)
            sum_hash[next_key_list]=1
            rumor=rumor+1

        else:
            next_key_list=key

    # actualizamos la matriz de rumores
    timerumor_matrix[count-1][0]=count
    timerumor_matrix[count-1][1]=rumor

    # si no se ha acabado el tiempo, seguimos contandolo, sino, se devuelve
    if count < 950:
        rec_search(next_key_list, count, net_hash, sum_hash, diam, timerumor_matrix, rumor)
    else:
        return timerumor_matrix

def drunk_walker(hash_matriz, diam):

    # a matrix with the times and the number of people that know the rumor
    timerumor_matrix = np.empty((950, 2))
    rumor=0

    # new hash with the nodes
    hash_prob={}
    for key in hash_matriz.keys():
        hash_prob[key]=0

    # cuantos paseos se da
    for i in range(1):
        counter=0 # busqueda recursiva
        random_key = random.choice(list(hash_matriz)) # empezamos por un punto aleatorio
        rec_search(random_key, counter, hash_matriz, hash_prob, diam, timerumor_matrix, rumor) # función de busqueda
    
    return timerumor_matrix

# OBTENEMOS VECES QUE HA ESTADO EN CADA NODO
timerumor_matrix = drunk_walker(hash_matriz, diameter)
print(timerumor_matrix[:,0])

plt.plot(timerumor_matrix[:,0], timerumor_matrix[:,1])
plt.xlabel('Time')
plt.ylabel('Number of people who know the rumor')
plt.show()


