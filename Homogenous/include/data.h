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
// #define EDGE_PATH "home/jvap2/Desktop/Code/FrogWild/Homogenous/Data/webGoogle.csv"
// #define CLUSTER_PATH "home/jvap2/Desktop/Code/FrogWild/Homogenous/Data/Cluster_Assignment.csv"

struct graph{
    struct vertex *point[NODES];
};

struct vertex{
    int end;
    struct vertex *next;
};

struct edge{
    int end, start;
};

__host__ void return_edge_list(string path, edge* arr);

__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size);

__global__ void bit_exclusive_scan(unsigned int* bits,unsigned int size);

__global__ void Sort_Cluster(unsigned int* cluster, unsigned int* vertex, unsigned int* table, unsigned int size,unsigned int iter);

__host__ void Org_Vertex_Helper(unsigned int* h_cluster, unsigned int* h_vertex, int size);

__global__ void Swap(unsigned int* cluster, unsigned int* vertex, unsigned int* table, unsigned int* table_2, unsigned  int size);