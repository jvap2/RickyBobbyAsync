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
        self.graph={}
        self.node_list=[]
        self.edge_list=[]
        self.node_degrees={}
    def generate_graph(self):
        for i in range(self.no_edges):
            self.add_edge()
    def add_edge(self):
        to, frm = self.generate_edge()
        self.graph[frm].append(to)
        self.edge_list.append((frm,to))
        self.node_degrees[frm]+=1
    def generate_edge(self, edge_list):
        to, frm = 

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











