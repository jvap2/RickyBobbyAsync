#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"
#define EDGES 5066842
#define NODES 855802
#define EDGE_PATH "../Data/webGoogle.csv"
#define CLUSTER_PATH "../Data/Cluster_Assignment.csv.csv"

__host__ void return_list(string path, int** arr);

__host__ void split_list(int** arr, int* subarr_1, int* subarr_2, int size);

__global__ void bit_exclusive_scan(int* bits);

__global__ void Sort_Cluster(int* cluster, int* vertex, int* table, int size, int iter);

__host__ void Org_Vertex_Helper(int* h_cluster, int* h_vertex, int size);