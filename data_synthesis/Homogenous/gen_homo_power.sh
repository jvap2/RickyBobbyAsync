#!/bin/bash

nodes=100000
m=50
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
