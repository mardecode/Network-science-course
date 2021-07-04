# ----------------------- Importando librerias ----------------------- #
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle
from pylab import *
# from scipy import stats as st
from sklearn import linear_model

def line():
	print("<><><><><><><><><><><><><><><><><><><><><><><><><><><>")

def drawG(G):
	nx.draw(G, with_labels=True)
	plt.show()


def getPickle(filename):
  infile = open(filename,'rb')
  data = pickle.load(infile)
  infile.close()
  return data

def createPickle(filename,filevalue):
  outfile = open(filename,'wb')
  pickle.dump(filevalue,outfile)
  outfile.close() 



def probability(degrees):
	sum = 0
	for i in degrees:
		sum += i[1]
	probability = [ i[1]/sum for i in degrees]
	return probability


def barbasiModel(m=1,size=10):
	barabasiG = nx.Graph()
	barabasiG.add_node(0)
	barabasiG.add_edges_from([(0, 0)])
	for i in range(1,size):
		degrees = barabasiG.degree()
		print(degrees)
		probabilities = probability(degrees)
		print(probabilities)
		print("--")
		barabasiG.add_node(i)
		for j in range(len(probabilities)):
			r = random.uniform(0,1)
			if(r<probabilities[j]):
				barabasiG.add_edges_from([(i,j)])
				break
	return barabasiG
# G = barbasiModel(size=10000)
G = nx.generators.barabasi_albert_graph(10000,1)


# createPickle("graph.pkl",G)
G = getPickle("graph.pkl")

degree_freq = nx.degree_histogram(G)


degrees = list(range(len(degree_freq)))


degree_freq__log = np.log(np.array(degree_freq))
degrees__log = np.log(np.array(degrees))
reg = linear_model.LinearRegression()

		
reg.fit(np.array(degrees__log).reshape(-1,1), np.array(degree_freq__log))

rango = np.arange(0,6,1)
regression = reg.predict(rango.reshape(-1,1))


# print(algo)


plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.plot(degrees, degree_freq,'bo')
# plt.plot(degrees__log, degree_freq__log,'bo')
# plt.plot(rango,regression)
plt.show()

# drawG(G)

# barbasiModel()





from pylab import * 
import networkx as nx
from scipy import stats as st
import math

n = 10000
ba = nx.barabasi_albert_graph(n,5)
pk = [float(x)/n for x in nx.degree_histogram(ba)]
domain = range(len(pk))

ccdf = [sum(pk[k:]) for k in domain]

logkdata = []
logFdata = []

prevF = ccdf[0]
for k in domain: 
	f = ccdf[k]
	if f != prevF:
		logkdata.append(math.log(k))
		logkdata.append(math.log(f))
		prevF = f

a,b,r,p,err = st.linregress(logkdata,logFdata)
plot(logkdata,logFdata,'o')

kmin,kmax = xlim()

plot([kmin,kmax],[a*kmin+b,a*kmax+b])
xlabel('log k')
ylabel('log F(k)')
show()
