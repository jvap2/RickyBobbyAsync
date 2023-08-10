#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../include/GPUErrors.h"
//Google
#define EDGES 5105039
#define NODES 875713
#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

struct graph{
    struct vertex *point[NODES];
};

struct vertex{
    int end;
    struct vertex *next;
};

struct edge{
    int end, start;
    int cluster;
};

__host__ void Check_Out_csv_edge(edge* edge_list);

__host__ void return_edge_list(string path, edge* arr);

__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size);

__global__ void bit_exclusive_scan(unsigned int* bits, unsigned int* bits_2, unsigned int size);

__global__ void Sort_Cluster(edge* edgelist, unsigned int* table, unsigned int size,unsigned int iter);

__host__ void Org_Vertex_Helper(edge* h_edge, int size);

__global__ void Swap(edge* edge_list, unsigned int* table, unsigned  int size, unsigned int iter);

__host__ graph *create_graph (edge *edges);

__global__ void Random_Edge_Placement(edge *edges, double rand_num);