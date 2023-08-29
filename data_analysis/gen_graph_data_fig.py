import matplotlib.pyplot as plt
import numpy as np
import os
import polars as pl
import sys


folder=os.getcwd()

df=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/replica_counts.csv"))

clusters=np.unique(df["num_clusters"].to_numpy())
no_nodes=np.unique(df["num_nodes"].to_numpy())

avg_rep=df["avg_replicas"].to_numpy()
avg_rep=avg_rep.reshape(len(no_nodes),len(clusters))
total_rep=df["total_replicas"].to_numpy()
total_rep=total_rep.reshape(len(no_nodes),len(clusters))

x=np.arange(len(clusters))
width =.1
cluster_labels = clusters.astype(str)
fig, (ax1)= plt.subplots(1,1)
fig.suptitle("Average Replicas vs. No. Clusters")
ax1.set_xlabel("No. Clusters")
ax1.set_ylabel("Average Replicas")
ax1.set_xticks(x+width,cluster_labels)
colors=["red","black","gray","blue","orange"]
align=["edge","edge","center","edge","edge"]
for i in range(-2,3):
    ax1.bar(x+(i+2)*width, avg_rep[:,i+2],width, label="No. Nodes = "+str(no_nodes[i+2]), color=colors[i+2], edgecolor="black", tick_label=cluster_labels)

plt.legend()
plt.savefig(os.path.join(os.path.dirname(folder),"Figures/avg_replicas.png"))
plt.show()

avg_rep_cluster= np.mean(avg_rep,axis=1)
print(avg_rep_cluster)  
fig, (ax1)= plt.subplots(1,1)
fig.suptitle("Average Replicas vs. No. Clusters")
ax1.set_xlabel("No. Clusters")
ax1.set_ylabel("Average Replicas")
ax1.plot(clusters,avg_rep_cluster)
ax1.set_xscale("log")
ax1.set_yscale("log")
plt.savefig(os.path.join(os.path.dirname(folder),"Figures/avg_replicas_cluster.png"))
plt.show()

