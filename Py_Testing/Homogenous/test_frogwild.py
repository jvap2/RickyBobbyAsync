import polars as pl
import os
import sys
import numpy as np


global clusters
clusters=16

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
        cluster_assign[Random_Edge_Placement(max(in_d[e[0]]+out_d[e[0]],in_d[e[1]]+in_d[e[0]]))].append([e[0],e[1]])
    return cluster_assign


def Random_Edge_Placement(i):
    cluster=i%clusters
    return cluster

def Gen_CSR(edge_list, no_nodes, no_edges):
    # edge_list=edge_list[edge_list[:,0].argsort()]
    src = np.zeros(shape=no_nodes+1)
    succ = np.zeros(shape=no_edges)
    for i,e in enumerate(edge_list):
        src[e[0]]+=1
        succ[i]=e[1]
    src_hold=np.zeros(shape=no_nodes+1)
    src_hold[1:]=src[:-1]
    src=src_hold
    for i in range(1,len(src)):
        src[i]+=src[i-1]
    return src, succ


def Gen_SubGraphs(cluster_assign):
    src_cluster={}
    succ_cluster={}
    local_src_vertices={}
    local_succ_vertices={}
    src_hash_table={}
    succ_hash_table={}
    for c in range(clusters):
        src_cluster[c]=[]
        succ_cluster[c]=[]
        local_src_vertices[c]=[]
        local_succ_vertices[c]=[]
        src_hash_table[c]={}
        succ_hash_table[c]={}
    
    '''We need to sift through the cluster assign and then add the src and succ to the respective clusters'''
    '''We need to extract all of the vertices within each cluster, local and ghost'''
    '''We are going to only save the src pointers from the source nodes in each cluster as a test'''
    for c in range(clusters):
        for e in cluster_assign[c]:
            local_src_vertices[c].append(e[0])
        '''Hold the values for the local vertices in global terms'''
        temp = np.unique(local_src_vertices[c])
        src_hash_table[c]=dict(zip(temp,range(len(temp))))
        src_cluster[c]=[0]*len(src_hash_table[c])  
        for v in local_src_vertices[c]:
            src_cluster[c][src_hash_table[c][v]]+=1
        for e in cluster_assign[c]:
            local_succ_vertices[c].append(e[1])
        '''Hold the values for the local vertices in global terms'''
        temp = np.unique(local_succ_vertices[c])
    return src_hash_table, src_cluster, succ_cluster
        

            



def init_1(no_nodes):
    init_pos=np.random.randint(0,no_nodes, size=no_nodes/10)
    return init_pos

def Gather(c,K,src_cluster,succ_cluster):
    pass

def Apply(c,K,src_cluster,succ_cluster):
    pass

def Scatter(c,K,src_cluster,succ_cluster):
    pass

def FrogWild(c,K,src_cluster,succ_cluster, no_nodes, iterations):
    init_pos=init_1(no_nodes)
    for i in range(iterations): 
        Gather(c,K,src_cluster,succ_cluster)
        Apply(c,K,src_cluster,succ_cluster)
        Scatter(c,K,src_cluster,succ_cluster)
    return c





df_edge=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net.csv"))
df_graph_data=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net_info.csv"))

no_nodes, no_edges = df_graph_data.get_column("No. Nodes")[0], df_graph_data.get_column("No. Edges ")[0]

edge_list = np.zeros(shape=(no_edges,2), dtype='int32')

edge_list[:,0],edge_list[:,1] = df_edge.get_column("from").to_numpy().tolist(), df_edge.get_column("to").to_numpy().tolist()

in_d, out_d =Get_Degree(edge_list, no_nodes)

src, succ = Gen_CSR(edge_list,no_nodes,no_edges)

cluster_assign = Degree_Cluster_Hash(clusters, edge_list, in_d, out_d)

'''How do we disperse the values for a random walk now?'''
'''Let us begin by making an array of arrays for each cluster with local_succ and local_ptr'''

sub_src, sub_succ = Gen_SubGraphs(cluster_assign)


'''We need to now commence the random walk'''
c={}
K={}
for n in range(no_nodes):
    c[n]=0
    K[n]=0











