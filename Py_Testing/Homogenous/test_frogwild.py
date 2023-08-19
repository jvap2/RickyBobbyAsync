import polars as pl
import os
import sys
import numpy as np
from math import modf,floor

global clusters
clusters=32

def Degree_Calculation(edge_list, no_nodes):
    out_degree=np.zeros(no_nodes)
    in_degree=np.zeros(no_nodes)
    for i in range(len(edge_list)):
        out_degree[edge_list[i][0]]+=1
        in_degree[edge_list[i][1]]+=1
    return out_degree,in_degree

def Random_Walk():
    pass

def Assign_Cluster(edge_list,no_edges, out_degree,in_degree):
    cluster_assign=np.zeros(no_edges)
    for i,e in enumerate(edge_list):
        ##Assign cluster based on the degree of the node in the edges
        cluster_assign[i]=Random_Edge_Placement(max(out_degree[e[0]]+in_degree[e[0]],in_degree[e[1]]+out_degree[e[1]]))

    return cluster_assign

def Generate_Sub_Graph(no_nodes,no_edges):
    pass



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











