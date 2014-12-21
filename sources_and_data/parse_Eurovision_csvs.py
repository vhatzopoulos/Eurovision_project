import networkx as nx
import csv
import numpy as np

filenames = ['Eurovision_2000.csv','Eurovision_2001.csv','Eurovision_2002.csv','Eurovision_2003.csv',\
'Eurovision_2004.csv','Eurovision_2005.csv','Eurovision_2006.csv','Eurovision_2007.csv','Eurovision_2008.csv',\
'Eurovision_2009.csv','Eurovision_2010.csv','Eurovision_2011.csv','Eurovision_2012.csv','Eurovision_2013.csv']

#filenames = filenames[3:]
winners = {filename:None for filename in filenames}
winners['Eurovision_2000.csv'] = 'Denmark'
winners['Eurovision_2001.csv'] = 'Estonia'
winners['Eurovision_2002.csv'] = 'Latvia'
winners['Eurovision_2003.csv'] = 'Turkey'
winners['Eurovision_2004.csv'] = 'Ukraine'
winners['Eurovision_2005.csv'] = 'Greece'
winners['Eurovision_2006.csv'] = 'Finland'
winners['Eurovision_2007.csv'] = 'Serbia'
winners['Eurovision_2008.csv'] = 'Russia'
winners['Eurovision_2009.csv'] = 'Norway'
winners['Eurovision_2010.csv'] = 'Germany'
winners['Eurovision_2011.csv'] = 'Azerbaijan'
winners['Eurovision_2012.csv'] = 'Sweden'
winners['Eurovision_2013.csv'] = 'Denmark'
g = nx.DiGraph()
allVotingCountries = []
noWinner = True

for filename in filenames:
	
	print filename
	
	votingCountries = []
	f = open(filename, 'rb')
	print filename
	reader = csv.reader(f, delimiter=',')
	numRows = 0
	
	for row in reader:
			
		 #extract the voting countries, i.e. all countries
		if numRows == 0:
			allVotingCountries.extend(row[2:])
			votingCountries.extend(row[2:])
			#for c in votingCountries:
			#	c = "\"" + c + "\""
			g.add_nodes_from(votingCountries)

		#found a participating country
		else:
			target = row[0]
			total = row[1]
			for i in range(len(votingCountries)):
				source = votingCountries[i]
				if len(row[i+2]) > 0: #if target voted for source, i.e non-empty field
					if target not in g[source]:

						g.add_edge(source,target,points=[])
						g[source][target]['points'].append(float(row[i+2]))
					else:

						g[source][target]['points'].append(float(row[i+2]))				
		numRows += 1
	
	#remove the winner
	if noWinner:
		winner = winners[filename]
		g.remove_node(winner)


#------------form the sum of edges
for e in g.edges(data=True):
	e[2]['points'] = np.sum(e[2]['points'])



#---------form--undirected by summing the directed edges weights

f = nx.Graph()
f.add_nodes_from(g)
for u in g:
	for v in g[u]:
		if u in g[v]:
			f.add_edge(u,v,points=g[u][v]['points']+g[v][u]['points'])
		else:
			f.add_edge(u,v,points=g[u][v]['points'])

nx.write_gpickle(f,"Eurovison_undirected_00_13_sum_points_no_winner")
nx.write_gpickle(g,"Eurovison_directed_00_13_sum_points_no_winner")

#----------HOME GROWN PAJEK INPUT FILE CREATION---------------------------------
"""	
	graphName = "graph" +"_" + filename +"_.net"
	f = open(graphName, 'wt')
	f.write('*Vertices')
	f.write(' %s' %(gd.number_of_nodes()))
	f.write('\n')
	#write vertices
	#map vertices to integers
	intToVertex={}
	i = 1
	for u in gd:
		f.write('%s'%i)
		f.write(' \"'+u+'\"')
		f.write('\n')
		intToVertex[u] = i
		i += 1
	
	#write edges
	f.write('*Arcs')
	f.write(' %s' %(gd.number_of_edges()))
	for e in gd.edges():
		f.write('\n')
		u = e[0]
		v = e[1]
		f.write('%s %s %s'%(intToVertex[u],intToVertex[v],gd[u][v]['transactions']))
	f.close()
#----------END---------------------------------	
"""