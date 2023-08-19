import polars as pl
import os
import sys
import numpy as np


global clusters
clusters=32

def Get_Degree(edge_list, no_nodes):
    in_degree = np.zeros(no_nodes)
    out_degree = np.zeros(no_nodes)
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

def Gen_CSR(edge_list, no_nodes, no_edges):
    edge_list=edge_list[edge_list[:,0].argsort()]
    src = np.zeros(shape=no_nodes)
    succ = np.zeros(shape=no_edges)
    for i,e in enumerate(edge_list):
        src[e[0]]+=1
        succ[i]=e[1]
    src_hold=np.zeros(shape=no_nodes)
    src_hold[1:]=src[:-1]
    src=src_hold
    for i in range(1,len(src)):
        src[i]+=src[i-1]
    return src, succ


def Gen_SubGraphs(succ,src,cluster_assign):
    src_cluster={}
    succ_cluster={}
    for c in range(clusters):
        src_cluster[c]=[]
        succ_cluster[c]=[]
    
    '''We need to sift through the cluster assign and then add the src and succ to the respective clusters'''
    for c in range(clusters):
        for v in cluster_assign[c]:
            print(v.keys())
            src_cluster[c].append(src[v.keys()])
            succ_cluster[c].append(succ[src[v.keys()[0]]:src[v.keys()[0]+1]])
    return src_cluster, succ_cluster


df_edge=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net.csv"))
df_graph_data=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net_info.csv"))

no_nodes, no_edges = df_graph_data.get_column("No. Nodes")[0], df_graph_data.get_column("No. Edges ")[0]

edge_list = np.zeros(shape=(no_edges,2), dtype='int32')

edge_list[:,0],edge_list[:,1] = df_edge.get_column("from").to_numpy().tolist(), df_edge.get_column("to").to_numpy().tolist()

in_d, out_d =Get_Degree(edge_list, no_nodes)

src, succ = Gen_CSR(edge_list,no_nodes,no_edges)

cluster_assign = Degree_Cluster_Hash(clusters, edge_list, in_d, out_d)

print(cluster_assign)
'''How do we disperse the values for a random walk now?'''
'''Let us begin by making an array of arrays for each cluster with local_succ and local_ptr'''

sub_src, sub_succ = Gen_SubGraphs(succ,src,cluster_assign)

print(sub_src)
print(sub_succ)

'''We need to now commence the random walk'''
c={}
K={}
for n in range(no_nodes):
    c[n]=0
    K[n]=0











