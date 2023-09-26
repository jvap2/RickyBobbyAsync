import os
import sys
import polars as pl

df_C = [pl.read_csv("Local_Cluster_Source_C.csv"), pl.read_csv("Local_Cluster_Successor_C.csv"), pl.read_csv("Local_Cluster_Unique_C.csv")]
df_py = [pl.read_csv("Local_Cluster_Source_python.csv"), pl.read_csv("Local_Cluster_Successor_python.csv"), pl.read_csv("Local_Cluster_Unique_python.csv")]

check=[0]*3
print("Source")
for c,py in zip(df_C[0],df_py[0]):
    for i,(c_row, py_row) in enumerate(zip(c,py)):
        if c_row==py_row:
            check[0]=1
        else:
            print("Row:",i,"C: ",c_row," Python: ",py_row)
            check[0]=0

print("Successor")
for c,py in zip(df_C[1],df_py[1]):
    for i,(c_row, py_row) in enumerate(zip(c,py)):
        if c_row==py_row:
            check[1]=1
        else:
            print("Row:",i,"C: ",c_row," Python: ",py_row)
            check[1]=0

print("Unique")
for c,py in zip(df_C[2],df_py[2]):
    for i,(c_row, py_row) in enumerate(zip(c,py)):
        if c_row==py_row:
            check[2]=1
        else:
            print("Row:",i,"C: ",c_row," Python: ",py_row)
            check[2]=0



if all(check):
    print("All files are equal")
else:
    print("Files are not equal")
    print(check)
    sys.exit(1)