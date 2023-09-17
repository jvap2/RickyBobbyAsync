import polars as pl
import os
import sys
import numpy as np
import cudf

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

cub_frog_acc = 0
cub_cug_acc = 0
frog_cug_acc = 0

k=200
for i in range(k):
    for j in range(k):
        if cub_vertex[i]==frog_vertex[j]:
            cub_frog_acc+=1
        if cub_vertex[i]==cug_vertex[j]:
            cub_cug_acc+=1
        if frog_vertex[i]==cug_vertex[j]:
            frog_cug_acc+=1

cub_frog_acc = cub_frog_acc/k
cub_cug_acc = cub_cug_acc/k
frog_cug_acc = frog_cug_acc/k

print("CUBLAS vs FROG: ", cub_frog_acc)
print("CUBLAS vs CUGRAPH: ", cub_cug_acc)
print("FROG vs CUGRAPH: ", frog_cug_acc)



