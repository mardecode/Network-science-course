import networkx as nx
import numpy as np
from networkx.algorithms.community import asyn_fluidc
from networkx.algorithms.community.label_propagation import asyn_lpa_communities


# ---------------------------------- Part b ---------------------------------- #

G = nx.read_graphml("net.graphml")

communities = asyn_fluidc(G,5)

# i = 0
# for community in communities:
#     f = open("community"+str(i)+".txt","w+")
#     group = ""
#     for node in community:
#       group+=node+"\n"
#     f.write(group)
#     f.close()
#     i+=1  

# ---------------------------------- Part c --------------------------------- #

communitiesNetworkx = asyn_lpa_communities(G)

cGephi = []
for community in communities:
  a = [ int(i) for i in community]
  a.sort(key=int)
  a = np.array(a)
  cGephi.append(a)


respGeral = True
for community in communitiesNetworkx:
  a = [ int(i) for i in community]
  a.sort(key=int)
  a = np.array(a)
  for j in range(5):
    if(cGephi[j][0] == a[0]):
      resp = np.array_equal(cGephi[j],a)
      if(resp!=True):
        respGeral = False
      print("Community "+ str(j) +": "+ str(resp))
print(respGeral)