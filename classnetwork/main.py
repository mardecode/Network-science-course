import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
import numpy as np

def averageDegree(graph):
  nodes = len(list(graph.nodes))
  links = len(graph.edges)
  resp = round((2*links)/nodes,2)
  return resp

def degreeDistribution(G):
  degree_freq = nx.degree_histogram(G)
  degrees = range(len(degree_freq))
  plt.xlabel('Degree')
  plt.ylabel('Frequency')
  plt.bar(degrees,degree_freq)
  plt.show()

def infoGraph(G):
  print("Average Degree: ",averageDegree(G))
  degreeDistribution(G)
  connectedComponents = nx.number_connected_components(G)
  print("Connected Components: ", connectedComponents)
  averageDistance = round(nx.average_shortest_path_length(G),2)
  print("Average Distance:", averageDistance)
  clusteringCoefficients = nx.clustering(G)
  cCoefficients = {}
  for k,c in clusteringCoefficients.items():
    cCoefficients[k] = round(c,2)
  print("Clustering coefficients: \n",cCoefficients)
  assortativity = round(nx.degree_assortativity_coefficient(G),2)
  print("Assortativity coefficient: ", assortativity)
# ---------------------------------------------------------------------------- #

G = nx.Graph()

data = pd.read_csv("class-network.tsv",sep='\t')
for i in range(13):
  filaname= 'H'+str(i)
  for j in range(len(data[filaname])):
    col = data[filaname][j]
    if(col == 1):
      hobbie = filaname
      person = data['H0'][j]
      G.add_edge(person,hobbie)

# ------------------------------ Bipartite graph ----------------------------- #
print("***Bipartite***")
# print([sorted((u, v)) for u, v in G.edges()])

X, Y = nx.bipartite.sets(G)
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(X) ) 
pos.update( (n, (2, i)) for i, n in enumerate(Y) ) 
# nx.draw(G, pos=pos,with_labels=True)
# plt.show()
# nx.write_gexf(G, "bipartite.gexf")
infoGraph(G)






# ------------------------------- Projection 1 - people ------------------------------- #
print("\n***Projection 1***")
peopleG =bipartite.projected_graph(G,X)
# print([sorted((u, v)) for u, v in peopleG.edges()])
# nx.write_gexf(peopleG, "peopleG.gexf")
# nx.draw(peopleG,with_labels=True)
# plt.show()
infoGraph(peopleG)


# ------------------------------- Projection 2 - hobbies ------------------------------- #
print("\n***Projection 2***")
hobbiesG =bipartite.projected_graph(G,Y)
# print([sorted((u, v)) for u, v in hobbiesG.edges()])
# nx.write_gexf(hobbiesG, "hobbiesG.gexf")
# nx.draw(hobbiesG,with_labels=True)
# plt.show()
infoGraph(hobbiesG)


