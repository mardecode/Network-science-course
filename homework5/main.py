import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt 

# ------------------------------- Reading net1 and net2 ------------------------------- #

net1 = pd.read_csv("net1.tsv",header=None,sep='\t')
graph1 = nx.from_pandas_edgelist(net1, 0, 1)

net2 = pd.read_csv("net2.tsv",header=None,sep=' ')
graph2 = nx.from_pandas_edgelist(net2, 0, 1)


# ---------------- get degree distribution of net1 and net2 ---------------- #

degree_freq = nx.degree_histogram(graph1)
degrees = range(len(degree_freq))

degree_freq2 = nx.degree_histogram(graph2)
degrees2 = range(len(degree_freq2))

# --------------------------- plot in log log scale -------------------------- #

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, sharey=True)
ax1.loglog(degrees, degree_freq,'bo')
ax1.set_title('Net 1')
ax1.set_xlabel('Degree')
ax1.set_ylabel('Frequency')

ax2.loglog(degrees2, degree_freq2,'bo')
ax2.set_xlabel('Degree')
ax2.set_title('Net 2')

fig.suptitle('Degree distribution in log log scale', fontsize=16)

plt.show()