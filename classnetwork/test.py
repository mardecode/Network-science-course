from os import PRIO_PGRP
from networkx.classes import graph
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite


G = nx.Graph()
G.add_edge(10,3)
G.add_edge(10,2)
G.add_edge(10,4)
G.add_edge(20,6)
G.add_edge(20,7)
G.add_edge(20,8)
G.add_edge(10,8)


print(G.nodes)




nx.draw(G,with_labels=True)
plt.show()


def degreeDistribution(G):
  degree_freq = nx.degree_histogram(G)
  degrees = range(len(degree_freq))
  print(degree_freq)
  plt.xlabel('Degree')
  plt.ylabel('Frequency')
  plt.bar(degrees,degree_freq)
  plt.show()

degreeDistribution(G)