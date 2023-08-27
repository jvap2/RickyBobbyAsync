#!/bin/bash

nodes=10000
m=250
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
