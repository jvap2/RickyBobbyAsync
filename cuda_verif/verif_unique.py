import numpy as np
import os
import polars as pl
import sys


folder=os.getcwd()
df=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm.csv"))
edges=df.height
clust=df["cluster"].to_numpy()
clusters = df["cluster"].unique().sort().to_numpy()
start = df["from"].to_numpy()
end = df["to"].to_numpy()
start_cluster=[ [] for _ in range(len(clusters)) ]
end_cluster=[ [] for _ in range(len(clusters)) ]

print(start_cluster)
print(end_cluster)
for i,val in enumerate(clust):
    start_cluster[val].append(start)
    end_cluster[val].append(end)