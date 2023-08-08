#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../include/GPUErrors.h"
//Google
#define EDGES 5105039
#define NODES 875713
#define EDGE_PATH "~/Desktop/Code/FrogWild/Homogenous/Data/webGoogle.csv"
#define CLUSTER_PATH "~/Desktop/Code/FrogWild/Homogenous/Data/Cluster_Assignment.csv"

__host__ void return_list(string path, int** arr);

__host__ void split_list(int** arr, int* subarr_1, int* subarr_2, int size);

__global__ void bit_exclusive_scan(int* bits, int* bit_2,int size);

__global__ void Sort_Cluster(int* cluster, int* vertex, int* table, int size, int iter);

__host__ void Org_Vertex_Helper(int* h_cluster, int* h_vertex, int size);

__global__ void Swap(int* cluster, int* vertex, int* table, int* table_2,  int size);