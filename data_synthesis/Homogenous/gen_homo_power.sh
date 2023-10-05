#!/bin/bash

nodes=30000
m=120
p=.6
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
