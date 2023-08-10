import os
import sys
import random
import numpy as np


folder=os.getcwd()[:-10]
Network = open(os.path.join(folder,"Data/rand/rand_net.csv"),"w") 


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