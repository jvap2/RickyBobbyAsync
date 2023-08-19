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

<<<<<<< HEAD
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
=======
>>>>>>> f1ca65146ff39d35ad6f6e7f052e603bb2bf2451

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











