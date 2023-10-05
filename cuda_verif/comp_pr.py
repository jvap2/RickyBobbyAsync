import polars as pl
import os
import sys
import numpy as np
import cudf
import random
import matplotlib.pyplot as plt

folder=os.getcwd()
df_cublas=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/CUBLAS_PR.csv"))
df_frog=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/C.csv"))
df_cugraph=cudf.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm_pagerank_py.csv"))
data=cudf.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Misc.csv"))
iterations=int(data["iterations"].to_numpy())
edges=int(data["edges"].to_numpy())
cluster=int(data["blocks"].to_numpy())
p_s=float(data["p_s"].to_numpy())



cub_vertex=df_cublas["Node"].to_numpy()
frog_vertex = df_frog["Node"].to_numpy()
cug_vertex = df_cugraph["vertex"].to_numpy()

cub_pr=df_cublas[" PageRank"].to_numpy()
frog_pr=df_frog["Count"].to_numpy()
frog_pr=frog_pr/np.sum(frog_pr)
cug_pr=df_cugraph["pagerank"].to_numpy()
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

k=100
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

exec_data = cudf.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Exec_Time_V1.csv"))
test_trial=exec_data.shape[0]
exec_data["acc_cub"].iloc[test_trial-1]=frog_cub_acc
exec_data["acc_cug"].iloc[test_trial-1]=frog_pr_acc
exec_data["top_num"].iloc[test_trial-1]=k
exec_data.to_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Exec_Time_V1.csv"),index=False)

folder=os.getcwd()
subfolder="Figures/influence_img/Data"+"_"+str(no_nodes)
save_folder=os.path.join(folder[:-10],subfolder)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

plt.figure()
plt.title("CUBLAS, FROG and CUGRAPH \nNo. Nodes={no_nodes}, No. Clusters={no_clusters}, Iterations={iter}, $p_s$={p}".format(no_nodes=no_nodes,no_clusters=cluster,iter=iterations,p=p_s))
plt.xlabel("Node")
plt.ylabel("PageRank")
plt.scatter(cub_vertex,cub_pr,label="CUBLAS", c="red")
plt.scatter(frog_vertex,frog_pr,label="FROG", c="black")
plt.scatter(cug_vertex,cug_pr,label="CUGRAPH", c="blue")
plt.legend()
file_name = "cub_frog_cug"+"_"+str(no_nodes)+"_"+str(cluster)+"_"+str(iterations)+"_"+str(p_s)+".png"
plt.savefig(os.path.join(save_folder,file_name))
plt.show()

## Show the cumulative accuracy

nodes=np.arange(no_nodes)
cub_sort=np.sort(cub_pr)
frog_sort=np.sort(frog_pr)
cug_sort=np.sort(cug_pr)

cub_cumsum=np.cumsum(cub_sort)
frog_cumsum=np.cumsum(frog_sort)
cug_cumsum=np.cumsum(cug_sort)

plt.figure()
plt.title("CUBLAS, FROG and CUGRAPH Cumulative Sum of Influence\nNo. Nodes={no_nodes}, No. Clusters={no_clusters}, Iterations={iter}, $p_s$={p}".format(no_nodes=no_nodes,no_clusters=cluster,iter=iterations,p=p_s))
plt.xlabel("Ranking")
plt.ylabel("Cumulative Influence")
plt.yscale("log")
plt.scatter(nodes,cub_cumsum,label="CUBLAS", c="red")
plt.scatter(nodes,frog_cumsum,label="FROG", c="black")
plt.scatter(nodes,cug_cumsum,label="CUGRAPH", c="blue")
plt.legend()
file_name = "cub_frog_cug_cumsum"+"_"+str(no_nodes)+"_"+str(cluster)+"_"+str(iterations)+"_"+str(p_s)+".png"
plt.savefig(os.path.join(save_folder,file_name))
plt.show()


labels=["FROG &\nCUBLAS","FROG &\nCUGRAPH","RAND &\nCUBLAS","RAND &\nCUGRAPH"]
acc=[frog_cub_acc,frog_pr_acc,rand_pr_acc, rand_cub_acc]
colors=["red","black","blue","gray"]
plt.figure()
plt.title("Accuracy of FrogWild! w.r.t cuBLAS and cuGraph (top {k})\nNo. Nodes={no_nodes}, No. Clusters={no_clusters}, Iterations={iter}, $p_s$={p}".format(k=k,no_nodes=no_nodes,no_clusters=cluster,iter=iterations,p=p_s))
plt.xlabel("Method")
plt.ylabel("Accuracy")
plt.bar(labels,acc,color=colors)
plt.grid()
file_name = "cub_frog_cug_acc"+"_"+str(no_nodes)+"_"+str(cluster)+"_"+str(iterations)+"_"+str(p_s)+".png"
plt.savefig(os.path.join(save_folder,file_name))
plt.show()






