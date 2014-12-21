import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


significance = [0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,\
0.00005,0.0001,0.0025,0.005,0.01,0.05,0.1,0.2]

#load the original digraph
gd = nx.read_gpickle("Eurovison_directed_00_13_sum_of_points")

edgeAttr = 'points'
#get the original graph statistics for normalization
n = float(gd.number_of_nodes())
e = float(gd.number_of_edges())
sumW = float(np.sum([gd.out_degree(weight=edgeAttr).values()]))
svns = []

fracValidatedNodes = []
fracValidatedEdges = []
fracValidatedWeight = []

for sig in sorted(significance):
	
	gName = "Eurovision_directed_00_13_bonferonni_"+str(sig)
	svn = nx.read_gpickle(gName)
	#svns.append(svn)

	#gName = "Eurovision_directed_00_13_bonferonni_"+str(sig)+"_.net"
	#svn = nx.read_pajek(gName)
	svns.append(svn)
	print gName,svn.number_of_nodes(),svn.number_of_edges()
	
for svn in svns:
	fracValidatedNodes.append(svn.number_of_nodes()/n)
	fracValidatedEdges.append(svn.number_of_edges()/e)
	
	sumWsvn = float(np.sum([svn.out_degree(weight=edgeAttr).values()]))
	fracValidatedWeight.append(sumWsvn/sumW)
	#components


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('significance')
ax.set_xlabel('N(svn)/N(g)')
ax.set_xscale('log')
ax.plot(significance,fracValidatedNodes,label='nodes')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('significance')
ax.set_xlabel('E(svn)/E(g)')
ax.set_xscale('log')
ax.plot(significance,fracValidatedEdges,label='edges')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('significance')
ax.set_xlabel('W(svn)/W(g)')
ax.plot(significance,fracValidatedWeight,label='weight')
ax.set_xscale('log')
plt.show()
