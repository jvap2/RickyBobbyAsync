import os
import sys
import random
import numpy as np
import polars as p


folder=os.getcwd()
Network = open(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"),"w") 


line = "from,to\n"
Network.write(line)

num_node = int(sys.argv[1])

np.random.seed(int(sys.argv[2]))

for house in range(num_node):
    num_used = int( np.random.exponential(35.0) )+1
    if num_used==0: num_used=int( np.random.exponential(40.0) )
    if num_used > 40: num_used = int( num_used / 20 )     # let's not generate too many edges
    p_system=np.zeros(shape=(num_used), dtype='int32')
    for i in range(0, num_used):
        p_system[i] = int( np.random.random() * num_node )
    p_system = np.unique(p_system)
    for i in range(len(p_system)):
        if p_system[i] != house:
            Network.write(str(house) + "," + str(p_system[i]) + "\n")

Network.close()

df=p.read_csv(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net.csv"))
edges=df.height
print(edges)
Network_Info = open(os.path.join(os.path.dirname(folder)[:-14],"Data/Homogenous/rand/rand_net_info.csv"), 'w')
Network_Info.write("No. Nodes,No. Edges \n")
Network_Info.write(str(sys.argv[1]) + "," + str(edges))

Network_Info.close()
