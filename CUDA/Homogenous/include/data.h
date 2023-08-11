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
// #define EDGE_PATH "../../Data/Homogenous/google/webGoogle.csv"
// #define CLUSTER_PATH "../../Data/Homogenous/google/Cluster_Assignment.csv"

#define EDGE_PATH "../../Data/Homogenous/rand/rand_net.csv"
#define GRAPH_DATA_PATH "../../Data/Homogenous/rand/rand_net_info.csv"
#define CLUSTER_PATH "../../Data/Homogenous/google/Cluster_Assignment.csv"
#define LIST_PATH "../../Data/Homogenous/google/list_check.csv"

struct graph{
    struct vertex *point[NODES];
};


struct vertex{
    unsigned long int end;
    struct vertex *next;
};

struct edge{
    unsigned long int end, start;
    unsigned long int cluster;
};

__host__ void Check_Out_csv_edge(edge* edge_list, int size);

__host__ void return_edge_list(string path, edge* arr);

__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size);

__global__ void bit_exclusive_scan(unsigned long int* bits, unsigned long int* bits_2, unsigned long int* bits_3, unsigned long int size);

__global__ void Sort_Cluster(edge* edgelist, unsigned long int* table, unsigned long int size,unsigned int iter);

__host__ void Org_Vertex_Helper(edge* h_edge, unsigned long int size);

__global__ void Swap(edge* edge_list, edge* edge_list_2, unsigned long int* table, unsigned long int* table_2, long int size, unsigned int iter);

__host__ graph *create_graph (edge *edges);

__global__ void Random_Edge_Placement(edge *edges, double rand_num);

__global__ void fin_exclusive_scan(unsigned long int* bits_3, unsigned long int size);

__host__ void get_graph_info(string path, unsigned long int* nodes, unsigned long int* edges);

__host__ void Check_Out_pref_sum(unsigned long int* list_1, unsigned long int* list_2, int size);

__host__ int getMax_cluster(edge* edge_list, int n);

__host__ void cpu_countSort(edge* arr, int n, int exp);

__host__ void cpu_radixsort(edge* arr, int n);

__global__ void copy_edge_list(edge* edge_1, edge* edge_2, unsigned long int size);