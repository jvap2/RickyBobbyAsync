import networkx as nx
import sys
import random
import numpy as np
import os
import polars as pl


folder=os.getcwd()

num_node = int(sys.argv[1])
seed = int(sys.argv[2])

G=nx.scale_free_graph(num_node,seed=seed)
nx.write_edgelist(G, os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"), delimiter=",", data=["from","to"])


df=pl.read_csv(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"))
edges=df.height
print(edges)
Network_Info = open(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net_info.csv"), 'w')
Network_Info.write("No. Nodes,No. Edges \n")
Network_Info.write(str(sys.argv[1]) + "," + str(edges))

Network_Info.close()