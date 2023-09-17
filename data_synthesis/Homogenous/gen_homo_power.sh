#!/bin/bash

nodes=10000
m=40
p=.5
seed=42

python3 netx_graph_gen.py $nodes $m $p $seed 
