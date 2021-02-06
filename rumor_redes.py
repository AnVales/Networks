# Implementation of Erdos-Renyi Model 
import numpy as np
from numpy import random 
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt

# VARIABLES
N = 20 # nodos
p = 0.5 # probabilidad de link
m = 10
k = 2

# RECORDATORIO DE FORMULAS
# average_degree = n*p
# average_number_links = p*n*(n-1)/2

# OBTENER LAS MATRICES
# erdos-renyi
# G1 = nx.generators.random_graphs.erdos_renyi_graph(N, p, seed=None, directed=False)
# matriz_random = nx.convert_matrix.to_numpy_array(G1)

# Barabási–Albert preferential attachment model
# G1 = nx.generators.random_graphs.barabasi_albert_graph(N, m, seed=None)
# matriz_random = nx.convert_matrix.to_numpy_array(G1)

# Watts–Strogatz small-world graph
G1 = nx.generators.random_graphs.watts_strogatz_graph(N, k, p, seed=None)
matriz_random = nx.convert_matrix.to_numpy_array(G1)

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

# # VEMOS INFORMACIÓN
print("Nº of edges", G1.number_of_edges())
print("Nº of nodes", G1.number_of_nodes())
# print("Diameter:", nx.algorithms.distance_measures.diameter(G1))
print("average degree connectivity of graph:", nx.algorithms.assortativity.average_degree_connectivity(G1))
print("Clustering:", nx.algorithms.approximation.clustering_coefficient.average_clustering(G1))
print("Number of connected components:", nx.number_connected_components(G1))
print("Is regular?", nx.algorithms.regular.is_regular(G1))
nx.draw(G1, node_color="r", node_size=100, with_labels=True)
plt.show()
# diameter=nx.algorithms.distance_measures.diameter(G1)

# RECORRAMOS LA RED
def rec_search(key, count, net_hash, sum_hash, timerumor_matrix, rumor):
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
        rec_search(next_key_list, count, net_hash, sum_hash, timerumor_matrix, rumor)
    else:
        return timerumor_matrix

def drunk_walker(hash_matriz):

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
        rec_search(random_key, counter, hash_matriz, hash_prob, timerumor_matrix, rumor) # función de busqueda
    
    return timerumor_matrix

# OBTENEMOS VECES QUE HA ESTADO EN CADA NODO
timerumor_matrix = drunk_walker(hash_matriz)
# print(timerumor_matrix[:,0])

plt.plot(timerumor_matrix[:,0], timerumor_matrix[:,1])
plt.xlabel('Time')
plt.ylabel('Number of people who know the rumor')
plt.ylim(0, 21)
plt.show()


