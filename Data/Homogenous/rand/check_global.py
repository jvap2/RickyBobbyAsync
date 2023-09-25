import polars as pl


pl_C_src = pl.read_csv("check/Global_Cluster_Source_C.csv")
pl_py_src = pl.read_csv("src_py.csv")
pl_C_succ = pl.read_csv("check/Global_Cluster_Successor_C.csv")
pl_py_succ = pl.read_csv("succ_py.csv")
pl_C_unq = pl.read_csv("check/Local_Cluster_Unique_C.csv")
pl_py_unq = pl.read_csv("unq_py.csv")
pl_C_unq_ptr=pl.read_csv("check/Local_Cluster_Unique_Ptr_Ctr_C.csv")
pl_py_unq_ptr=pl.read_csv("unq_ptr_py.csv")

pl_C_src = pl_C_src["src"].to_numpy()
pl_py_src = pl_py_src["src"].to_numpy()
pl_C_succ = pl_C_succ["succ"].to_numpy()
pl_py_succ = pl_py_succ["succ"].to_numpy()
pl_C_unq = pl_C_unq["unq"].to_numpy()
pl_py_unq = pl_py_unq["unq"].to_numpy()
pl_C_unq_ptr=pl_C_unq_ptr["unq_ptr"].to_numpy()
pl_py_unq_ptr=pl_py_unq_ptr["unq_ptr"].to_numpy()

check=[0]*4
print("Source")
for p,c in zip(pl_py_src,pl_C_src):
    if p==c:
        check[0]=1
    else:
        check[0]=0
        break
print("Successor")
for p,c in zip(pl_py_succ,pl_C_succ):
    if p==c:
        check[1]=1
    else:
        check[1]=0
        break
print("Unique")
for i,(p,c) in enumerate(zip(pl_py_unq,pl_C_unq)):
    if p==c:
        check[3]=1
    else:
        check[3]=0
        print(i,p,c)
print("Unique Ptr")
for i, (p,c) in enumerate(zip(pl_py_unq_ptr,pl_C_unq_ptr)):
    if p==c:
        check[2]=1
    else:
        check[2]=0
        print(i,p,c)

if all(check):
    print("All files are equal")
else:
    print("Files are not equal")
    print(check)