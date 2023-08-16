import cugraph as cg
import cudf
import polars as pl
import os
import sys

df=cudf.read_csv(os.path.join(os.getcwd()[:-18],"Data/Homogenous/rand/rand_net.csv"),dtype=["int32", "int32"])
g= cg.from_cudf_edgelist(df,source="from",destination="to")

pr_vert = cg.pagerank(G=g)

pr_vert.to_csv(os.path.join(os.getcwd()[:-18],"Data/Homogenous/rand/pr_res/homo_pr_scores.csv"), sep=",",index=False)




