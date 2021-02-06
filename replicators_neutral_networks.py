import networkx as nx
from networkx.algorithms.centrality import eigenvector
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms import approximation
from numpy.matrixlib.defmatrix import matrix

# erdos_renyi: 20 nodes y p=0.3
nodos = 20
p = 0.3

G1 = nx.generators.random_graphs.erdos_renyi_graph(
    nodos, p, seed=None, directed=False)

G1_dict = nx.convert.to_dict_of_dicts(G1)
G1_array = nx.convert_matrix.to_numpy_array(G1)
G1_matrix = nx.convert_matrix.to_numpy_matrix(G1)
# nx.draw(G1, node_color="r", node_size=140, with_labels=True)
# plt.show()

# print("Nº of edges", G1.number_of_edges())
# print("Nº of nodes", G1.number_of_nodes())
# print("Diameter:", nx.algorithms.distance_measures.diameter(G1))
# print("average degree connectivity of graph:",
#       nx.algorithms.assortativity.average_degree_connectivity(G1))
# print("Clustering:",
#       nx.algorithms.approximation.clustering_coefficient.average_clustering(G1))
# print("Number of connected components:", nx.number_connected_components(G1))
# print("Is regular?", nx.algorithms.regular.is_regular(G1))

# preferencial attachment: 20 nodes, two coupled nodes
nodos = 20
m = 2

# G2 = nx.generators.random_graphs.barabasi_albert_graph(nodos, m, seed=42)

# G2_dict = nx.convert.to_dict_of_dicts(G2)
# G2_dict = nx.convert.to_dict_of_dicts(G2)
# G2_array = nx.convert_matrix.to_numpy_array(G2)
# G2_matrix = nx.convert_matrix.to_numpy_matrix(G2)
# nx.draw(G2, node_color="r", node_size=140, with_labels=True)
# plt.show()

# print("Nº of edges", G2.number_of_edges())
# print("Nº of nodes", G2.number_of_nodes())
# print("Diameter:", nx.algorithms.distance_measures.diameter(G2))
# print("average degree connectivity of graph:", nx.algorithms.assortativity.average_degree_connectivity(G2))
# print("Clustering:", nx.algorithms.approximation.clustering_coefficient.average_clustering(G2))
# print("Number of connected components:", nx.number_connected_components(G2))
# print("Is regular?", nx.algorithms.regular.is_regular(G2))

# modelizamos Population dynamics on neutral networks
poblacion = 100
f = 2
mu = 0.2
A = 4
l = 5


def several_generations(lamb, u, n, time, matriz_n, tiempo_max, matriz_result):
    actualizacion = {}
    time = time + 1

    for i in range(len(u)):
        eigenvalue = lamb[i]
        eigenvector = u[:, i]

        primera_multi = n*eigenvector
        segunda_multi = primera_multi*eigenvalue
        tercera_multi = eigenvector*segunda_multi

        actualizacion[i] = tercera_multi

    suma_final = np.zeros((actualizacion[0].shape))
    for i in range(len(u)):
        suma_final = actualizacion[i] + suma_final

    for i in range(len(u)):
        if suma_final[i] < 0:
            suma_final[i] = 0

    n = suma_final.reshape(1, 20)
    matriz_n[time, :] = n

    total_pop = np.sum(n)

    prop = n/total_pop

    # result = np.zeros(20)
    # for i in range(len(prop)):
    #     eigenvector_i = u[1, i]
    #     prop_i = lamb[i]
    #     result[i] = prop_i/eigenvector_i

    matriz_result[time, :] = prop
    maxValue = -999999
    minValue = 999999

    for i in range(len(prop)):
        if maxValue < prop[0,i]:
            maxValue = prop[0,i]
        if minValue > prop[0,i]:
            minValue = prop[0,i]
    if maxValue<0.076 and minValue>0.024:
        print(time)
        # print(prop)
        return matriz_result
    else:
        several_generations(lamb, u, n, time, matriz_n, tiempo_max, matriz_result)


    # if time < tiempo_max-1:
    #     # print (n)
    #     several_generations(lamb, u, n, time, matriz_n, tiempo_max, matriz_result)
    # else:
    #     return matriz_result


# hacemos la matriz M, hacemos el siguiente n(t) y esperamos a que en todos los n(t) tengamos 5 en cada nodo y guardamos los tiempos a los que esto pasa
def dynamic_nn(poblacion, nodos, f, mu, A, l, G):

    # condiciones iniciales de n(0)
    # n = np.zeros(nodos)
    n = np.ones(nodos)*(poblacion/nodos)
    # n[0] = poblacion
    tiempo_max = 901
    time = 0

    matriz_n = np.zeros((tiempo_max, nodos))
    matriz_result = np.zeros((tiempo_max, nodos))
    matriz_n[time, :] = n

    # matriz M
    I = np.identity(len(G))
    primero = f*(1-mu)*I
    segundo_numerador = f*mu
    segundo_denominador = (A-1)*l
    segundo_division = segundo_numerador/segundo_denominador
    segundo = segundo_division*G
    M = primero + segundo

    # actualizamos n(t)
    # lamb eigenvalues (shape 20)
    # u eigenvectors (shape 20*20)
    lamb, u = np.linalg.eig(M)

    several_generations(lamb, u, n, time, matriz_n, tiempo_max, matriz_result)

    return matriz_result


evolucion_n = dynamic_nn(poblacion, nodos, f, mu, A, l, G1_matrix)
# print(evolucion_n)

# time_vector = np.linspace(0, 900, 901)

# for i in range(nodos):
#     plt.plot(time_vector, evolucion_n[:, i])
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.show()


# plt.plot(time_vector, evolucion_n[:,0])
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.show()
