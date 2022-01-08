import networkx as nx 
import argparse
import numpy as np
import os
import pickle
import sys
import dgl

sys.setrecursionlimit(5500000)


root = '../resources/dataset/c/'

def load_data(part):
	data_path = os.path.join(root, part,'blocks.pkl')
	with open(data_path, 'rb') as f:
		dataset = pickle.load(f)
	return dataset

def dfs_visit_recursively(g,node,nodes_color,edges_to_be_removed):

	nodes_color[node] = 1
	nodes_order = list(g.successors(node))
	# nodes_order = np.random.permutation(nodes_order)
	for child in nodes_order:
		if nodes_color[child] == 0:
				dfs_visit_recursively(g,child,nodes_color,edges_to_be_removed)
		elif nodes_color[child] == 1:
			edges_to_be_removed.append((node,child))

	nodes_color[node] = 2

def dfs_remove_back_edges(adjacency_list):
	'''
	0: white, not visited 
	1: grey, being visited
	2: black, already visited
	'''
	# g = nx.read_edgelist(graph_file,create_using = nx.DiGraph(),nodetype = nodetype)
	g = nx.DiGraph()
	g.add_edges_from(adjacency_list)

	nodes_color = {}
	edges_to_be_removed = []
	for node in list(g.nodes()):
		nodes_color[node] = 0

	nodes_order = list(g.nodes())
	# nodes_order = np.random.permutation(nodes_order)
	num_dfs = 0
	for node in nodes_order:
		if nodes_color[node] == 0:
			num_dfs += 1
			dfs_visit_recursively(g,node,nodes_color,edges_to_be_removed)
	#print("number of nodes to start dfs: %d" % num_dfs)
	#print("number of back edges: %d" % len(edges_to_be_removed))
	g.remove_edges_from(edges_to_be_removed)
	return list(g.edges), edges_to_be_removed


if __name__ == '__main__':
	# dataset=load_data('test')
	adjacency_list =[[1,5],[1,8],[2,4],[2,5],[2,6],[3,5],[4,5],[4,6],[4,7],[5,7],[6,5],[6,8],[7,4],[7,5],[7,6]]
	# temp = []
	new_adjacency_list, loop_edges = dfs_remove_back_edges(adjacency_list)
	print(loop_edges)
	g = dgl.DGLGraph(list(new_adjacency_list))
	print(new_adjacency_list)
	print(list(dgl.topological_nodes_generator(g)))
