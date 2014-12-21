import networkx as nx
#from operator import itemgetter
import csv
import matplotlib.pyplot as plt
import numpy as np
import myMathTools as mmt
import myGraphAlgorithms as mga
reload(mga)
import operator


gd = nx.read_gpickle("Eurovison_directed_00_13_sum_of_points")
gu = nx.read_gpickle("Eurovison_undirected_00_13_sum_of_points")
#gd = nx.read_gpickle("Eurovison_directed_03_13_all_points")
#gu = nx.read_gpickle("Eurovison_undirected_03_13_all_points")
g = gd
edgeAttr = 'points'
significance = [0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,\
0.00005,0.0001,0.001,0.0025,0.005,0.01,0.05,0.1,0.2]

correcting = True
if nx.is_directed(g):
	isDirected = True
else:
	isDirected = False
	
fracValidatedNodes = []
fracValidatedLinks = []
fracValidatedWeight = []


print sorted(significance)
#-----------link-------validation-----------------
#for all levels of significance
for sig in sorted(significance):

	edgesToRemove = []
	gval = g.copy()
	n = g.number_of_nodes()
	edges = g.number_of_edges()
	weightSum = np.sum([e[2][edgeAttr] for e in g.edges(data=True)])
	
	print "validating with significance " + str(sig)
	print "nodes " + str(n) + " edges " + str(edges)
	
	
	validatedLinks = mga.validate_over_represented(g,edgeAttr,sig,correcting,isDirected)
	for e in g.edges():
		if e not in validatedLinks:
			edgesToRemove.append(e)
	
	gval.remove_edges_from(edgesToRemove)
	gval.remove_nodes_from(nx.isolates(gval))
	
	fracValidatedNodes.append(gval.number_of_nodes()/float(n))
	fracValidatedLinks.append(len(validatedLinks)/float(edges))
	
	validatedWeightSum = np.sum([gval[u][v][edgeAttr] for (u,v) in validatedLinks])
	fracValidatedWeight.append(validatedWeightSum/float(weightSum))
			
			
	print "validated nodes: " + str(gval.number_of_nodes()/float(n)) 
	print "validated links: " + str(len(validatedLinks)/float(edges))
	print "validated weight: " + str(validatedWeightSum/float(weightSum))
	
	#write the vaidated network
	gName = "Eurovision_directed_00_13_bonferonni_"+str(sig)+"_.net"
	mga.write_pajek(gval,edgeAttr,gName,isDirected)
	gName = "Eurovision_directed_00_13_bonferonni_"+str(sig)
	nx.write_gpickle(gval,gName)
	
	
			
			
#----------file--writting
f = open('validated_Eurovision_graph_bonferroni.txt','a')
for i in range(len(fracValidatedWeight)):
	strfractionNodes = str(fracValidatedNodes[i])
	strfractionEdges = str(fracValidatedEdges[i])
	strfractionWeight = str(fracValidatedWeight[i])
	f.write('%s,%s,%s' %(strfractionNodes[0:6],strfractionEdges[0:6],strfractionWeight[0:6]))
	f.write('\n')
f.close()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(significance,fracValidatedNodes,label='nodes')
ax.plot(significance,fracValidatedEdges,label='edges')
ax.plot(significance,fracValidatedWeight,label='weights')
ax.legend()
plt.show()
			
			
	




#gName = "Eurovision_directed_00_13_SVN_bonferonni_alpha_00025.net"
#mga.write_pajek(g,edgeAttr,gName,isDirected=True)
#print "done writing SVN"





 







