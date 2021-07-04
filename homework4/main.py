# ----------------------- Importando librerias ----------------------- #
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

# ----------------------- Average degree  ----------------------- #
def averageDegree(graph):
    nodes = len(list(graph.nodes))
    links = len(graph.edges)
    return (2*links)/nodes

# ----------------------- largest component  ----------------------- #
def maxLenComponent(graph):
    return len(max(nx.connected_components(graph), key=len))

# ----------------------- random Graph  ----------------------- #
def randomGraph(p,nodes,max_nodes):
    bk = False
    randomG = nx.Graph()
    randomG.add_nodes_from(range(1,nodes+1))
    component_size = maxLenComponent(randomG)
    rep = 0
    while(component_size<=max_nodes):
        rep += 1
        for i in range(1,nodes+1):
            for j in range(i+1,nodes+1):
                u = random.uniform(0, 1)
                if(u>p):
                    randomG.add_edges_from([(i, j)])
                    component_size = maxLenComponent(randomG)
                    if(component_size>=max_nodes):
                        bk = True
                        break
            if(bk):
                break
    return randomG

# ----------------------- Experiment  ----------------------- #
def experiment(p,nodes,max_nodes,repeat):
    average_degrees = []
    for i in range(repeat):
        randomGr = randomGraph(p=p,nodes=nodes,max_nodes=max_nodes)  
        average_degrees.append(averageDegree(randomGr))
    return average_degrees


p = random.random()
degress = experiment(p,nodes=1000,max_nodes=501,repeat=50)

plt.hist(degress,bins=20)
plt.show()