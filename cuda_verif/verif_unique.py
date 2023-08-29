import numpy as np
import os
import polars as pl
import sys


folder=os.getcwd()
df=pl.read_csv(os.path.join(os.path.dirname(folder),"Data/Homogenous/rand/Cluster_Assignment_Norm.csv"))

clusters = df["cluster"].unique().sort().to_numpy()
print(clusters)
