import polars as pl
import numpy as np
import os
import sys

def inclusive_sum(arr_1,arr_2):
    '''
    Parameters:
    arr_1: histogram values
    arr_2: array to use for inclusive sum
    '''
    arr_2[0]=0
    for i in range(1,len(arr_2)):
        arr_2[i]=arr_2[i-1]+arr_1[i-1]
    return arr_2

# Get the path to the data
df=pl.read_csv("rand_net.csv")
df_info=pl.read_csv("rand_net_info.csv")
df_clust=pl.read_csv("Cluster_Assignment_Norm.csv")
cluster_counts=df_clust["cluster"].value_counts().to_numpy()[:,1].tolist()
new_clust_cnt=[0]*(len(cluster_counts)+1)
new_clust_cnt=inclusive_sum(cluster_counts,new_clust_cnt)
print(new_clust_cnt)
frm_unq=[[]]*len(cluster_counts)
to_unq=[[]]*len(cluster_counts)
unq_nodes=[[]]*len(cluster_counts)
unq_ptr=[0]*len(cluster_counts)
for i in range(len(cluster_counts)):
    frm_unq[i]=df_clust["from"][new_clust_cnt[i]:new_clust_cnt[i+1]].to_list()
    to_unq[i]=df_clust["to"][new_clust_cnt[i]:new_clust_cnt[i+1]].to_list()
    frm_unq[i]=list(set(frm_unq[i]))
    to_unq[i]=list(set(to_unq[i]))
    unq_nodes[i]=list(set(frm_unq[i]+to_unq[i]))
    unq_ptr[i]=len(unq_nodes[i])
unq_ptr_final=[0]*(len(cluster_counts)+1)
unq_ptr_final=inclusive_sum(unq_ptr,unq_ptr_final)
print(unq_ptr_final)
unq_nodes_final=[]
for i in range(len(unq_nodes)):
    unq_nodes_final+=unq_nodes[i]
# Find the unqiue nodes in each cluster
# unq_nodes=np.array(unq_nodes)
# unq_nodes=unq_nodes.flatten()
nodes=df_info["No. Nodes"][0]
print("Nodes: ",nodes)
src=[0]*nodes
succ=[]

frm = df["from"]
to = df["to"]

for f,t in zip(frm,to):
    src[f]+=1
    succ.append(t)




src_fin = [0]*(nodes+1)
src_fin = inclusive_sum(src,src_fin)
#Export src and succ data
pl.DataFrame({"src":src_fin}).write_csv("src_py.csv")
pl.DataFrame({"succ":succ}).write_csv("succ_py.csv")
pl.DataFrame({"unq":unq_nodes_final}).write_csv("unq_py.csv")
pl.DataFrame({"unq_ptr":unq_ptr_final}).write_csv("unq_ptr_py.csv")
     
