import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

net = pd.read_csv("net1000-005.tsv",header=None,sep='\t')
graph = nx.from_pandas_edgelist(net, 0, 1)

# --------------------------------- Section a -------------------------------- #

sub_graph = net.to_numpy()
sub_graph = sub_graph[np.random.choice(len(sub_graph), size=100, replace=False)]
nodes = []
with open('nodes', 'w') as f:
    for i in range(100):
        f.write(str(i)+" "+str(sub_graph[i][0])+"\n")
        nodes.append(sub_graph[i][0])
#nodes

# --------------------------------- section b -------------------------------- #

distances = []
def shortestPath(i,j):
    path = nx.bidirectional_shortest_path(graph,i,j)
    resp = ""
    distances.append(len(path)-1)
    for i in path:
        resp += str(i)+" "
    return resp

with open('paths', 'w') as f:
    for i in range(50):
        path = shortestPath(nodes[2*i],nodes[2*i+1])
        f.write(path+"\n")
#distances

# --------------------------------- section c -------------------------------- #
distances = np.array(distances)
plt.xlabel('distances')
plt.ylabel('number of pairs')
plt.hist(distances)
plt.show()