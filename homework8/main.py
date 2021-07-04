# ----------------------- Importando librerias ----------------------- #

from networkx.classes.function import degree
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import math
from sklearn import linear_model
import itertools

def m_t(t):  
  return math.floor(math.sqrt(t))

def network(nodes):
  G = nx.Graph()   
  G.add_node(0)
  rep = [0]
  t=0
  while t < nodes:
    m = m_t(t)
    choose = random.choices(rep,k=m) 
    G.add_edges_from([(t,choose[i]) for i in range(m)]) 
    rep += choose
    rep += [t] * m
    t += 1
  return G

G = network(10000)


# ----------------------------- b) log-log scale ----------------------------- #
degree_freq = nx.degree_histogram(G)
degrees = range(len(degree_freq))

plt.loglog(degrees,degree_freq,'bo')
plt.bar(degrees,degree_freq)
plt.show()

# ----------------------------- c)  straight line ---------------------------- #
dee_log = []
freq_log = []

for i in range(len(degrees)):
  if(degrees[i] != 0 and degree_freq[i] != 0):
    dee_log.append(math.log10(degrees[i]))
    freq_log.append(math.log10(degree_freq[i]))

reg = linear_model.LinearRegression()
reg.fit(np.array(dee_log[10:220]).reshape(-1,1),np.array(freq_log[10:220]))
print("gamma :",reg.coef_)
  
rango = np.arange(1,3,0.1)
regression = reg.predict(rango.reshape(-1,1))

plt.xlim(0,3.3)
plt.ylim(-0.1,2.7)

plt.plot(dee_log,freq_log,'bo')
plt.plot(rango,regression,'r')

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.show()

