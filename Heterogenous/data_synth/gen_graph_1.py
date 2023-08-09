import os
import sys
import random
import numpy as np

Network=open("net.csv","w")


line = ["#delimieter: ,\n","#columns: house,station\n","#types: UINT,UINT\n"]

for l in line:
    Network.write(l)

num_p_system = int(sys.argv[2])
num_houses = int(sys.argv[1])

np.random.see(int(sys.argv[3]))

for house in range(num_houses):
    num_used = int( np.random.exponential(20.0) )
    if num_used > 20: num_used = int( num_used / 10 )     # let's not generate too many edges

    for i in range(0, num_used):
        p_system = int( np.random.random() * num_p_system )
        Network.write(str(house) + "," + str(p_system) + "\n")