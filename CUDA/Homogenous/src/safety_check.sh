#!/bin/bash
arg=0
make frog
./bin/Frog
cd ../../cuda_verif
python3 verif_unique.py
cd ../Data/Homogenous/rand/check
python3 check_C_py_local.py
if[$? -eq 0]
then
    echo "The results are correct"
else
    arg = -1

cd ../../../../CUDA/Homogenous
make Frogwild
./bin/Frogwild $arg
