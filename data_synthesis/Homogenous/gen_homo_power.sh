#!/bin/bash

nodes=2000
m=80
p=.6
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
