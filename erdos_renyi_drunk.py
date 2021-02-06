# Implementation of Erdos-Renyi Model 
import numpy as np
from numpy import random 
import networkx as nx
from networkx.algorithms import approximation
import matplotlib.pyplot as plt

# VARIABLES
N=1000 # nodos
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
nx.draw(G, node_color="r", node_size=50, with_labels=False)
plt.show()
diameter=nx.algorithms.distance_measures.diameter(G)

# RECORRAMOS LA RED
def rec_search(key, count, steps, net_hash, sum_hash, diam):
    count = count + 1
    steps = steps + 1
    values = net_hash[key]
    next_key_list = random.choice(list(values))
    if steps > diam+2:
        sum_hash[next_key_list]=sum_hash[next_key_list]+1
        steps = 0
    if count < 50:
        rec_search(next_key_list, count, steps, net_hash, sum_hash, diam)

def drunk_walker(hash_matriz, diam):

    # new hash with the nodes
    hash_prob={}
    for key in hash_matriz.keys():
        hash_prob[key]=0

    # cuantos paseos se da
    for i in range(125):
        counter=0 # busqueda recursiva
        timesteps=0 # pasos de guardado
        random_key = random.choice(list(hash_matriz)) # empezamos por un punto aleatorio
        rec_search(random_key, counter, timesteps, hash_matriz, hash_prob, diam) # función de busqueda
    
    return hash_prob

# OBTENEMOS VECES QUE HA ESTADO EN CADA NODO
final_hash_prob = drunk_walker(hash_matriz, diameter)
print(final_hash_prob)

# MIRAMOS CUANTOS SITIOS HAN SIDO VISITADOS

def check_visits(final_hash):
    with open('borracho_100', 'w') as f:
        visit=0
        prob_hash={}
        for i in final_hash.values():
            visit=visit+i
        for key in final_hash.keys():
            value=final_hash[key]
            value=int(value)
            prob_hash[key]=(value/visit)*100
            f.write(str(key))
            f.write('\t')
            f.write(str((value/visit)*100))
            f.write('\n')
    f.close()

    return prob_hash



visitas=check_visits(final_hash_prob)
print(visitas)




