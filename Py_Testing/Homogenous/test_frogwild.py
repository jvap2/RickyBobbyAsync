import polars as pl
import os
import sys
import numpy as np
from math import modf,floor

global clusters
clusters=32


class Dir_Graph:
    def __init__(self, no_edges):
        self.no_nodes=0
        self.no_edges=no_edges
        self.src=[]
        self.succ=[]
        self.idx=[]
        self.out_degree=[]
    def gen_CSR(self, edge_list):
        '''Here, we want to read the edge_list in and generate the CSR representation of the graph
        The attributes of the class that will be worked on are the following:
        (1) src: this points to the starting index of the successors to a node
        (2) succ: this is the list of successors to a node
        (3) idx: this is the index of the node
        (4) out_degree: this is the out degree of the node
        '''
        idx,self.out_degree=np.unique(edge_list[:,0]) ##This provides a list of the unique outgoing values in the graph, and their respective counts
        self.no_nodes=len(idx)
        self.src=np.zeros(self.no_nodes+1)



def Random_Edge_Placement(i):
    cluster=i%clusters
    return cluster



df_edge=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net.csv"))
df_graph_data=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net_info.csv"))

no_nodes, no_edges = df_graph_data.get_column("No. Nodes")[0], df_graph_data.get_column("No. Edges ")[0]

to, frm = df_edge.get_column("to").to_numpy(), df_edge.get_column("from")
num_out_neigh = frm.value_counts().sort(by='from').to_numpy()



g={}
count=0
for c,n in enumerate(num_out_neigh[:,0]):
    g[n]=[]
    for i in range(num_out_neigh[c,1]):
        g[n].append(to[count])
        count+=1


cluster_assign={}
for c in range(clusters):
    cluster_assign[c]=[]

count=0
for key in g.keys():
    for val in g[key]:
        cluster_assign[Random_Edge_Placement(count)].append({key:val})
        count+=1

'''We need to now commence the random walk'''
c={}
K={}
for n in range(no_nodes):
    c[n]=0
    K[n]=0











