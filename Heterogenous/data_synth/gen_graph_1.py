import os
import sys
import random
import numpy as np


folder=os.getcwd()[:-10]

num_p_system = int(sys.argv[2])
num_houses = int(sys.argv[1])

np.random.seed(int(sys.argv[3]))
count=0

line = ["#delimieter: ,\n","#columns: house,station\n","#types: UINT,UINT\n", "house,station\n"]

with open(os.path.join(folder,"Data/net.csv"),"w") as fp:
    if count<len(line):
        fp.write(line[count])
        count+=1
    else:
        for house in range(num_houses):
            num_used = int( np.random.exponential(20.0) )
            if num_used > 20: num_used = int( num_used / 10 )     # let's not generate too many edges

            for i in range(0, num_used):
                p_system = int( np.random.random() * num_p_system )
                fp.write(str(house) + "," + str(p_system) + "\n")