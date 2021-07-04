# libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# graph
net = pd.read_csv("net1000-005.tsv",header=None,sep='\t')
graph = nx.from_pandas_edgelist(net, 0, 1)
nx.draw(graph, with_labels=True)
plt.show()

#frequency
degree_freq = nx.degree_histogram(graph)
degrees = range(len(degree_freq))
plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.bar(degrees,degree_freq)
plt.show()

#average degree
nodes = len(list(graph.nodes))
links = len(graph.edges)
average_degree = (2*links)/nodes
print("Average degree: ",average_degree)