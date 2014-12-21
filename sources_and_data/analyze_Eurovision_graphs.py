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
#gd = nx.read_gpickle("Eurovison_directed_00_13_sum_points_no_winner")
#gu = nx.read_gpickle("Eurovison_undirected_00_13_sum_points_no_winner")
#gd = nx.read_gpickle("Eurovison_directed_03_13_all_points")
#gu = nx.read_gpickle("Eurovison_undirected_03_13_all_points")
g = gu
edgeAttr = 'points'
sig = 0.005,0.005,0.01,0.05,0.1
significance = [0.005,0.001,0.01,0.05,0.1]
correct = False

if nx.is_directed(g):
	isDirected = True
else:
	isDirected = False

for sig in significance:

	#gName = "Eurovision_undirected_00_13_SVN_bonferroni_alpha_" + str(sig)
	gName = "Eurovision_undirected_00_13_SVN_no_correction_alpha_" + str(sig)
	print "validating " + gName
	#-----------link-------validation-----------------
	valLinks = mga.validate_over_represented_fast(g,edgeAttr,sig,correct,isDirected)
	edgesToRemove = []

	gval = g.copy()

	for e in g.edges():
		if e not in valLinks:
			edgesToRemove.append(e)

	gval.remove_edges_from(edgesToRemove)
	gval.remove_nodes_from(nx.isolates(gval))
	
	
	nx.write_gpickle(gval,gName)
	gName += ".net"
	print gName
	mga.write_pajek(gval,edgeAttr,gName,isDirected=True)
	#print "done writing SVN"
	#input()

#print "valLinks",valLinks
#input()
#-----------clustering-------coefficient-----------------
"""
vertexToInt = mga.make_vertex_to_int(g)
cCycle,cMiddle,cIn,cOut = mga.directed_weighted_clustering_coefficient(g,vertexToInt,edgeAttr)
meancCycle = np.mean(cCycle.values())
meancMiddle = np.mean(cMiddle.values())
meancIn = np.mean(cIn.values())
meanCout = np.mean(cOut.values())
print "original graph"
print meancCycle,meancMiddle,meancIn,meanCout
#original graph
#0.0898350708451 0.099743138862 0.0885589444904 0.107522582512

vertexToInt = mga.make_vertex_to_int(gval)
cCycle,cMiddle,cIn,cOut = mga.directed_weighted_clustering_coefficient(gval,vertexToInt,edgeAttr)
meancCycle = np.mean(cCycle.values())
meancMiddle = np.mean(cMiddle.values())
meancIn = np.mean(cIn.values())
meanCout = np.mean(cOut.values())
print "validated graph"
print meancCycle,meancMiddle,meancIn,meanCout
#validated graph
#0.106339048163 0.141062141145 0.103077057413 0.148233465078
"""
#sort cc dicts by value
#sorted_x = sorted(cCycle.iteritems(), key=operator.itemgetter(1))

#calculate--- the-- weight--- distribution
"""
yearsInSample = 11 #03-13
yearsInSample = 14 #00-13
maxPointsPerYear = 12
maxPointsPerPairPerYear = 2*maxPointsPerYear
#the maximum number of points that a pair can have voted for each other
#in the sample (maybe correcte for final participation)
maxPointsPerPairInSample = maxPointsPerPairPerYear*yearsInSample 

pointCounts = [0]*maxPointsPerPairInSample
for e in g.edges(data=True):
	points = int(e[2]['points'])
	pointCounts[points] += 1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(pointCounts,'*-k')
ax.set_xlabel('sum of points between country pair in' + str(yearsInSample) + " years")
ax.set_ylabel('count')
#ax.set_xscale('log')
plt.show()
input()
"""



 

#filter--edges--by--percentile
"""
per=95
#e.g. mean value, high coefficient of variation(volatile voting)
edgesToRemove = []

#edges =  g.edges(data=True)
points = [e[2]['points'] for e in g.edges(data=True)]
percentileCuttoff = np.percentile(points,per)
if g == gu:
	print str(per)+" percentile for undirected graph is " + str(percentileCuttoff)
else:
	print str(per)+" percentile for directed graph is " + str(percentileCuttoff)
input()

for e in g.edges(data=True):

	#if edgeCV[(e[0],e[1])] >= cvCuttoff or edgeCV[(e[0],e[1])] == 0:
	#if np.mean(e[2]['points']) < meanCuttoff:
	#if np.sum(e[2]['points']) < sumCuttoff:
	#if np.sum(e[2]['points']) < pcSumCuttoff:
	if e[2]['points'] < percentileCuttoff:
		edgesToRemove.append(e)
		print len(edgesToRemove)

g.remove_edges_from(edgesToRemove)
"""
#other-cuttofs
"""
numYears = len(filenames)
cuttoff = 10
meanCuttoff = 11
sumCuttoff = numYears*4
cvCuttoff = 0.1
"""

#--write--to-pajek--for--visualtions--in--mapequation.org
"""
if g == gu:
	nx.write_pajek(g,"undirected_graph_above_"+str(per)+"th_percentile_03_13.net")
else:
	nx.write_pajek(g,"directed_graph_above_"+str(per)+"th_percentile_03_13.net")
"""


#----------END---------------------------------	

"""
#nx.write_pajek(g,"Eurovison_graph_06_13_high_points.net")
#nx.write_pajek(g,"Eurovison_graph_05_13_high_points.net")
#nx.write_pajek(g,"Eurovison_graph_05_13_mean_points_greater_than_10.net")
#nx.write_pajek(g,"Eurovison_graph_06_13_mean_points.net")
#nx.write_pajek(g,"Eurovison_graph_06_13_low_CV.net")
#nx.write_pajek(g,"Eurovison_graph_03_13_mean_greater_than_11.net")
#nx.write_pajek(g,"Eurovison_graph_03_13_low_cv.net")
#nx.write_pajek(g,"Eurovison_graph_03_13_sum_greater_than_44.net")
#nx.write_pajek(g,"Eurovison_graph_00_13_top_20_percent.net")
#nx.write_pajek(g,"test.net")
"""

#----------coefficient of variation
"""
edgeCV = {}
for e in g.edges(data=True):
	#assign mean points to each edge
	#e[2]['points'] = np.mean(e[2]['points'])
	#print e[0],e[1],e[2]['points']
	edgeCV[(e[0],e[1])] = np.std(e[2]['points'])/np.mean(e[2]['points'])
plt.hist(edgeCV.values(),bins=40)
plt.show()	
"""
"""
means = []
cvs = []
for e in g.edges(data=True):
	
	#print e
	means.append(np.mean(e[2]['points']))
	cvs.append(np.std(e[2]['points'])/np.mean(e[2]['points']))
	#e[2]['points'] = np.mean(e[2]['points'])
	e[2]['points'] = np.sum(e[2]['points'])
	
edges = g.edges(data=True)
cv=[edgeCV[(e[0],e[1])] for e in edges]
s=[e[2]['points'] for e in edges]
print np.corrcoef(cv,s)

plt.plot(means,cvs,'*')
#plt.hist([e[2]['points'] for e in g.edges(data=True)])
plt.show()	
"""

#------------plot--sorted--edge--weights--TO--DO:Distribution
"""
points =[e[2]['points'] for e in g.edges(data=True)]
fig=plt.figure()
ax=fig.add_subplot(111)
#ax.plot(sorted(points,reverse=True))
ax.plot(sorted([np.mean(point) for point in points],reverse=True))
plt.show()
"""

