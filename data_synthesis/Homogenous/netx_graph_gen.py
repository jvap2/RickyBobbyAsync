import networkx as nx
import sys
import random
import numpy as np
import os
import polars as pl


folder=os.getcwd()

num_node = int(sys.argv[1])
m = int(sys.argv[2])
p = float(sys.argv[3])
seed = int(sys.argv[4])

G=nx.powerlaw_cluster_graph(num_node, m, p, seed=seed)
Network_Info = open(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"), 'w')
Network_Info.write("from,to\n")
# nx.write_edgelist(G, os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"), delimiter=",", data=False)
for edge in G.edges:
    Network_Info.write(str(edge[0]) + "," + str(edge[1]) + "\n")


df=pl.read_csv(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"))
edges=df.height
print(edges)
Network_Info = open(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net_info.csv"), 'w')
Network_Info.write("No. Nodes,No. Edges \n")
Network_Info.write(str(sys.argv[1]) + "," + str(edges))

Network_Info.close()