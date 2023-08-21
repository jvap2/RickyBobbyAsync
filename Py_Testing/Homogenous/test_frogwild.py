import polars as pl
import os
import sys
import numpy as np


global clusters
clusters=4

def Get_Degree(edge_list, no_nodes):
    in_degree = np.zeros(no_nodes)
    out_degree = np.zeros(no_nodes)
    for e in edge_list:
        in_degree[e[0]]+=1
        out_degree[e[1]]+=1
    return in_degree, out_degree

def Degree_Cluster_Hash(clusters, edge_list, in_d, out_d):
    cluster_assign={}
    '''We need to find the number of replicas or mirrors are present'''
    replicas= {}
    for c in range(len(in_d)):
        replicas[c]=[]
    for c in range(clusters):
        cluster_assign[c]=[]
    for e in edge_list:
        cluster=Random_Edge_Placement(max(out_d[e[0]]+in_d[e[0]],out_d[e[1]]+in_d[e[1]]))
        cluster_assign[cluster].append([e[0],e[1]])
        replicas[e[0]].append(cluster)
        replicas[e[1]].append(cluster)
    for i,r in enumerate(replicas.values()):
        temp=set(r)
        replicas[i]=list(temp)
    return cluster_assign, replicas


def Random_Edge_Placement(i):
    cluster=int(i%clusters)
    return cluster

def Gen_CSR(edge_list, no_nodes, no_edges):
    # edge_list=edge_list[edge_list[:,0].argsort()]
    src = np.zeros(shape=no_nodes+1, dtype='int32')
    succ = np.zeros(shape=no_edges, dtype='int32')
    for i,e in enumerate(edge_list):
        src[e[0]]+=1
        succ[i]=e[1]
    src_hold=np.zeros(shape=no_nodes+1, dtype='int32')
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
    hash_table={}
    for c in range(clusters):
        src_cluster[c]=[]
        succ_cluster[c]=[]
        local_src_vertices[c]=[]
        local_succ_vertices[c]=[]
        hash_table[c]={}
    '''We need to sift through the cluster assign and then add the src and succ to the respective clusters'''
    '''We need to extract all of the vertices within each cluster, local and ghost'''
    '''We are going to only save the src pointers from the source nodes in each cluster as a test'''
    for c in range(clusters):
        for e in cluster_assign[c]:
            local_src_vertices[c].append(e[0])
            local_succ_vertices[c].append(e[1])
        '''Hold the values for the local vertices in global terms'''
        temp = np.concatenate((local_src_vertices[c],local_succ_vertices[c]), axis=None)
        src_unq_len = len(np.unique(local_src_vertices[c]))
        lst_nodes,idx = np.unique(temp, return_index=True)
        lst_nodes = [temp[i] for i in sorted(idx)]
        src_cluster[c]=[0]*(src_unq_len+1)
        hash_table[c]=dict(zip(lst_nodes,range(len(lst_nodes))))
        src_cluster[c][0]=0  
        for v in local_src_vertices[c]:
            src_cluster[c][hash_table[c][v]+1]+=1
        for i in range(1,len(src_cluster[c])):
            src_cluster[c][i]+=src_cluster[c][i-1]
        '''Hold the values for the local vertices in global terms'''
        succ_cluster[c]=[0]*len(local_succ_vertices[c])
        for i,v in enumerate(local_succ_vertices[c]):
            succ_cluster[c][i]=hash_table[c][v]
    return hash_table, src_cluster, succ_cluster
        

            
'''How do we define a frontier of active nodes?'''


def init_1(no_nodes):
    init_pos=np.random.randint(0,no_nodes, size=int(no_nodes/2))
    return init_pos

def Gather(c,K):
    for clust in range(clusters):
        for i, val in enumerate(K[clust]):
                '''We need to simulate the dying of frogs'''
                if K[clust][i]>0:
                    liv = np.random.randint(0,1)
                    if liv<=.15:
                        K[clust][i]-=1
                c[clust][i]+=val

def Apply(c,K,src_cluster,succ_cluster):
    pass

def Scatter(c,K,src_cluster,succ_cluster):
    pass

def FrogWild(c,K,src, succ, src_cluster,succ_cluster, succ_hash, no_nodes, iterations):
    init_pos=np.unique(init_1(no_nodes))
    '''We need activate the initial nodes'''
    active_clusters={}
    for clust in range(clusters):
        active_clusters[clust]=0
    for i in init_pos:
        for clust in range(clusters):
            if i in succ_hash[clust]:
                K[clust][succ_hash[clust][i]]=+1
                active_clusters[clust]=1
    '''The frontier has now been defined, now we need to first gather the values from the frontier, and evaluate where the frogs will jump to next'''
    '''This is now the scatter portion of the algorithm'''
    for clust in range(clusters):
        '''We are going to iterate throught the clusters and then find the active nodes'''
        if active_clusters[clust]==1:
            '''We have nodes here which are active'''
            for i, val in enumerate(K[clust]):
                '''We have an active node'''
                c[clust][i]+=val
                '''We have incremented the counter for the node in the cluster'''
                '''We need to find successors within the cluster'''
                if i<len(src_cluster[clust])-1:
                    '''We have successors locally'''
                    num_local_neigh = src_cluster[clust][i+1]-src_cluster[clust][i]
                    location = np.random.randint(0,num_local_neigh, size=val)
                    '''We have a location for the successor'''
                    for l in location:
                        K[clust][succ_cluster[clust][src_cluster[clust][i]+l]]+=1
                else:
                    '''Successors are in a different cluster'''
                    '''We need to use the hash table to locate the global address and send the value to the correct cluster'''
                    '''We need to find the global address of the node'''
                    global_address = succ_hash[clust][i]
                    '''We need to find the cluster(s) that the node is in'''
                    non_local_succ = succ[src[global_address]:src[global_address+1]]
                    if len(non_local_succ)>1:
                        location = np.random.randint(non_local_succ[0], non_local_succ[-1], size=val)
                        for clust_2 in range(clusters):
                            for l in location:
                                if l in succ_hash[clust_2]:
                                    K[clust_2][succ_cluster[clust_2][succ_hash[clust_2][l]]]+=1
                                    active_clusters[clust_2]=1
                    else:
                        location = non_local_succ[0]
                        for clust_2 in range(clusters):
                                if location in succ_hash[clust_2]:
                                    K[clust_2][succ_cluster[clust_2][succ_hash[clust_2][location]]]+=1
                                    active_clusters[clust_2]=1
                    '''We have a location for the successor'''
                    '''We need to find the cluster that the successor is in'''

    for i in range(iterations): 
        Gather(c,K)
        Apply(c,K,src_cluster,succ_cluster)
        Scatter(c,K,src_cluster,succ_cluster)
    return c



if __name__ == "__main__":

    df_edge=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net.csv"))
    df_graph_data=pl.read_csv(os.path.join(os.getcwd()[:-21],"Data/Homogenous/rand/rand_net_info.csv"))

    no_nodes, no_edges = df_graph_data.get_column("No. Nodes")[0], df_graph_data.get_column("No. Edges ")[0]

    edge_list = np.zeros(shape=(no_edges,2), dtype='int32')

    edge_list[:,0],edge_list[:,1] = df_edge.get_column("from").to_numpy().tolist(), df_edge.get_column("to").to_numpy().tolist()

    in_d, out_d =Get_Degree(edge_list, no_nodes)

    src, succ = Gen_CSR(edge_list,no_nodes,no_edges)

    cluster_assign, num_replicas = Degree_Cluster_Hash(clusters, edge_list, in_d, out_d)
    # print(cluster_assign)
    # print(num_replicas)

    '''How do we disperse the values for a random walk now?'''
    '''Let us begin by making an array of arrays for each cluster with local_succ and local_ptr'''

    hash_table, sub_src, sub_succ = Gen_SubGraphs(cluster_assign)
    

    print("SUCC HASH TABLE")
    print('----------------')
    print(hash_table)
    print("SUB SRC ARRAY")
    print('----------------')
    print(sub_src)
    print("SUB SUCC ARRAY")
    print('----------------')
    print(sub_succ)


    '''We need to now commence the random walk'''
    clust_c={}
    clust_k={}
    '''I think it actually may be worthwhile to save c and K in csr format (Maybe, Maybe not)
    Need to look further into how frontiers are saved
    '''
    for c in range(clusters):
        clust_c[c]=[0]*len(hash_table[c])
        clust_k[c]=[0]*len(hash_table[c])

    # print("CLUST C")
    # print('----------------')
    # print(clust_c)
    # print("CLUST K")
    # print('----------------')
    # print(clust_k)



    FrogWild(clust_c,clust_k,src, succ, sub_src,sub_succ,hash_table,no_nodes,10)
