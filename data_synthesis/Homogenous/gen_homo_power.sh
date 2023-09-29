#!/bin/bash

nodes=20000
m=60
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
