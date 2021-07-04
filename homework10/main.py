import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import random

net = pd.read_csv("netA.csv",header=None,sep='\t')
G = nx.from_pandas_edgelist(net, 0, 1)

def n_max_component(G):
  largest_cc = len(max(nx.connected_components(G), key=len))
  return largest_cc

incre =  0.05
f = 0.0
size = len(G.nodes)
f_vec = []
p_vec = []

while f<1:
  newG = nx.from_pandas_edgelist(net, 0, 1)
  for n in G:
    nrandom = random.uniform(0,1)
    if (nrandom<=f):
      newG.remove_node(n)
  prob = n_max_component(newG)/len(newG.nodes)
  f_vec.append(f)
  p_vec.append(prob)

  f+=incre

plt.scatter(f_vec,p_vec)
plt.show()


