import polars as pl
import os
import sys
import numpy as np
import cudf
import random

folder=os.getcwd()
df_cublas=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/CUBLAS_PR.csv"))
df_frog=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/C.csv"))
df_cugraph=cudf.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm_pagerank_py.csv"))


cub_vertex=df_cublas["Node"].to_numpy()
frog_vertex = df_frog["Node"].to_numpy()
cug_vertex = df_cugraph["vertex"].to_numpy()
print(cub_vertex.shape)
print(frog_vertex.shape)
print(cug_vertex.shape)

no_nodes=df_cublas.height

min_cos_sim = (1.0*no_nodes+2.0)/(2.0*no_nodes+1.0)

cub_frog_sim = np.dot(cub_vertex,frog_vertex)/(np.linalg.norm(cub_vertex,ord=2)*np.linalg.norm(frog_vertex,ord=2))
cub_cug_sim = np.dot(cub_vertex,cug_vertex)/(np.linalg.norm(cub_vertex,ord=2)*np.linalg.norm(cug_vertex,ord=2))
frog_cug_sim = np.dot(frog_vertex,cug_vertex)/(np.linalg.norm(frog_vertex,ord=2)*np.linalg.norm(cug_vertex,ord=2))


print("CUBLAS vs FROG: ", cub_frog_sim)
print("CUBLAS vs CUGRAPH: ", cub_cug_sim)
print("FROG vs CUGRAPH: ", frog_cug_sim)

## Compare the accuraccy of the three methods
rand_chance = np.arange(no_nodes)
np.random.shuffle(rand_chance)




frog_cub_acc=0
frog_pr_acc=0
rand_cub_acc=0
rand_pr_acc=0
pr_cub_acc=0

k=int(.1*no_nodes)
for i in range(k):
    for j in range(k):
        if(frog_vertex[i]==cug_vertex[j]):
            frog_pr_acc+=1
            # print("FROG vs CUGRAPH: ", frog_vertex[i],cug_vertex[j])
        if(frog_vertex[i]==cub_vertex[j]):
            frog_cub_acc+=1
            # print("FROG vs CUBLAS: ", frog_vertex[i],cub_vertex[j])
        if(rand_chance[i]==cug_vertex[j]):
            rand_pr_acc+=1
            # print("RAND vs CUGRAPH: ", rand_chance[i],cug_vertex[j])
        if(rand_chance[i]==cub_vertex[j]):
            rand_cub_acc+=1
            # print("RAND vs CUBLAS: ", rand_chance[i],cub_vertex[j])
        if(cug_vertex[i]==cub_vertex[j]):
            pr_cub_acc+=1
            # print("CUGRAPH vs CUBLAS: ", cug_vertex[i],cub_vertex[j])
        

frog_pr_acc=frog_pr_acc/k
frog_cub_acc=frog_cub_acc/k
rand_pr_acc=rand_pr_acc/k
rand_cub_acc=rand_cub_acc/k
pr_cub_acc=pr_cub_acc/k

print("FROG vs CUBLAS accuracy: {:.4f}, RAND vs CUBLAS accuracy: {:.4f}".format(frog_cub_acc,rand_cub_acc))
print("FROG vs CUGRAPH accuracy: {:.4f}, RAND vs CUGRAPH accuracy: {:.4f}".format(frog_pr_acc,rand_pr_acc))
print("CUGRAPH vs CUBLAS accuracy: {:.4f}".format(pr_cub_acc))



