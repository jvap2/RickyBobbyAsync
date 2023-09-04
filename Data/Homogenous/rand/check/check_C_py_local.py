import os
import sys
import polars as pl

df_C = [pl.read_csv("Local_Cluster_Source_C.csv"), pl.read_csv("Local_Cluster_Successor_C.csv"), pl.read_csv("Local_Cluster_Unique_C.csv")]
df_py = [pl.read_csv("Local_Cluster_Source_python.csv"), pl.read_csv("Local_Cluster_Successor_python.csv"), pl.read_csv("Local_Cluster_Unique_python.csv")]

check=[]

for i, df in enumerate(df_C):
    check.append(df.frame_equal(df_py[i]))


if all(check):
    print("All files are equal")
else:
    print("Files are not equal")
    print(check)
    sys.exit(1)