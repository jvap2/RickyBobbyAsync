#!/bin/bash

nodes=80000
m=90
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
