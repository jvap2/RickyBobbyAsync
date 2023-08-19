import polars as pl
import os
import sys
import numpy as np
from math import modf,floor

global clusters
clusters=32

def Get_Degree(edge_list, no_nodes):
    in_degree = np.zeroes(no_nodes)
    out_degree = np.zeroes(no_nodes)
    for e in edge_list:
        in_degree[e[0]]+=1
        out_degree[e[1]]+=1
    return in_degree, out_degree

def Degree_Cluster_Hash(clusters, edge_list, in_d, out_d):
    cluster_assign={}
    for c in range(clusters):
        cluster_assign[c]=[]
    for e in edge_list:
        cluster_assign[Random_Edge_Placement(max(in_d[e[0]]+out_d[e[0]],in_d[e[1]]+in_d[e[0]]))].append({e[0]:e[1]})
    return cluster_assign


def Random_Edge_Placement(i):
    cluster=i%clusters
    return cluster



df_edge=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net.csv"))
df_graph_data=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net_info.csv"))

no_nodes, no_edges = df_graph_data.get_column("No. Nodes")[0], df_graph_data.get_column("No. Edges ")[0]

edge_list = [df_edge.get_column("to").to_numpy(), df_edge.get_column("from").to_numpy()]

in_d, out_d =Get_Degree(edge_list, no_nodes)

cluster_assign = Degree_Cluster_Hash(clusters, edge_list, in_d, out_d)

print(cluster_assign)



'''We need to now commence the random walk'''
c={}
K={}
for n in range(no_nodes):
    c[n]=0
    K[n]=0











