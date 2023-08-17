import polars as pl
import os
import sys
import numpy as np
from math import modf

global clusters
clusters=32

def Random_Edge_Placement(edge, r):
    cluster_list=[]
    for i,_ in enumerate(edge):
        cluster_list.append((modf(i*r)[0])*clusters)
    return cluster_list



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







