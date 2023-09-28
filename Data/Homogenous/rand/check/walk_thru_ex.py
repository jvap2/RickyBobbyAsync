import polars as pl
import numpy as np
import os

unq=pl.read_csv("Local_Cluster_Unique_C.csv")
unq_ptr=pl.read_csv("Local_Cluster_Unique_Ptr_Ctr_C.csv")
local_src = pl.read_csv("Local_Cluster_Source_C.csv")
local_succ = pl.read_csv("Local_Cluster_Successor_C.csv")