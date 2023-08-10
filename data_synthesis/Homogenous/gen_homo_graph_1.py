import os
import sys
import random
import numpy as np
import polars as p


folder=os.getcwd()
Network = open(os.path.join(os.path.dirname(folder),"Data/rand/rand_net.csv"),"w") 


line = ["#delimieter: ,\n","#types: UINT,UINT\n", "from,to\n"]

for l in line:
    Network.write(l)

num_node = int(sys.argv[1])

np.random.seed(int(sys.argv[2]))

for house in range(num_node):
    num_used = int( np.random.exponential(20.0) )
    if num_used==0: num_used=int( np.random.exponential(20.0) )
    if num_used > 20: num_used = int( num_used / 10 )     # let's not generate too many edges

    for i in range(0, num_used):
        p_system = int( np.random.random() * num_node )
        Network.write(str(house) + "," + str(p_system) + "\n")

df=p.read_csv(os.path.join(folder,"Data/rand/rand_net.csv"))
edges=df.height
print(edges)
Network_Info = open(os.path.join(folder,"Data/rand/rand_net_info.csv"), 'w')
Network_Info.write("No. Nodes, No. Edges \n")
Network_Info.write(str(sys.argv[1]) + "," + str(edges))

