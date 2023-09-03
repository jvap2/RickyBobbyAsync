import numpy as np
import os
import polars as pl
import sys

def merge_sequential(arr_1, arr_2):
    i=0
    j=0
    k=0
    arr_1.sort()
    arr_2.sort()
    fin = [None]*(len(arr_1)+len(arr_2))
    while i < len(arr_1) and j < len(arr_2):
        if arr_1[i] <= arr_2[j]:
            fin[k]=arr_1[i]
            i+=1
            k+=1
        else:
            fin[k]=arr_2[j]
            j+=1
            k+=1
    while i < len(arr_1):
        fin[k]=arr_1[i]
        i+=1
        k+=1
    while j < len(arr_2):
        fin[k]=arr_2[j]
        j+=1
        k+=1
    return fin


folder=os.getcwd()
df=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm.csv"))
print(df.groupby("cluster", maintain_order=True).agg(pl.count()))
edges=df.height
clust=df["cluster"].to_numpy()
clusters = df["cluster"].unique().sort().to_numpy()
start = df["from"].to_numpy()
end = df["to"].to_numpy()
start_cluster=[ [] for _ in range(len(clusters)) ]
end_cluster=[ [] for _ in range(len(clusters)) ]
merge_unq = [ [] for _ in range(len(clusters)) ]


for i,val in enumerate(clust):
    start_cluster[val].append(start[i])
    end_cluster[val].append(end[i])

for val in clusters:
    merge_unq[val]=merge_sequential(start_cluster[val],end_cluster[val])
    merge_unq[val]=list(set(merge_unq[val]))

ren_unq = [ [] for _ in range(len(clusters)) ]
for val in clusters:
    ren_unq[val]=[i for i in range(len(merge_unq[val]))]

renum_src = [ [] for _ in range(len(clusters)) ]
renum_succ = [ [] for _ in range(len(clusters)) ]

for i in range(len(clusters)):
    renum_src[i]=[0]*len(ren_unq[i])
    print(len(merge_unq[i]))
    renum_succ[i]=[0]*len(start_cluster[i])
