#!/bin/bash

nodes=1000
m=80
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
