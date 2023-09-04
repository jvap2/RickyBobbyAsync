import numpy as np
import os
import polars as pl
import sys
from itertools import chain


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
renum_src_temp = [ [] for _ in range(len(clusters)) ]
renum_succ = [ [] for _ in range(len(clusters)) ]

for i in range(len(clusters)):
    renum_src[i]=[0]*(len(ren_unq[i])+1)
    renum_src_temp[i]=[0]*(len(ren_unq[i])+1)
    renum_succ[i]=[0]*len(start_cluster[i])

renum_start = [ [] for _ in range(len(clusters)) ]
renum_end = [ [] for _ in range(len(clusters)) ]

for i,val in enumerate(clust):
    renum_start[val].append(ren_unq[val][merge_unq[val].index(start[i])])
    renum_end[val].append(ren_unq[val][merge_unq[val].index(end[i])])



for i,val in enumerate(clusters):
    for s in renum_start[val]:
        renum_src[val][s]+=1

for val in clusters:
    renum_src[val]=inclusive_sum(renum_src[val],renum_src_temp[val])

for i,val in enumerate(clust):
    for j in range(len(renum_src[val])-1):
        for k in range(renum_src[val][j], renum_src[val][j+1]):
            renum_succ[val][k]=renum_end[val][k] 


df_succ = pl.DataFrame({"cluster":clust, "succ":list(chain.from_iterable(renum_succ))}, schema={"cluster":pl.Int64, "succ":pl.Int64})
df_src = pl.DataFrame({"cluster":[], "src":[]}, schema={"cluster":pl.Int64, "src":pl.Int64})
df_unq = pl.DataFrame({"cluster":[], "unq":[]}, schema={"cluster":pl.Int64, "unq":pl.Int64})

for i,val in enumerate(clusters):
    df_unq=df_unq.extend(pl.DataFrame({"cluster":[val]*len(merge_unq[val]), "unq":merge_unq[val]}))
    df_src=df_src.extend(pl.DataFrame({"cluster":[val]*len(renum_src[val]), "src":renum_src[val]}))


df_succ.write_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Local_Cluster_Successor_python.csv"))
df_src.write_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Local_Cluster_Source_python.csv"))
df_unq.write_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/check/Local_Cluster_Unique_python.csv"))