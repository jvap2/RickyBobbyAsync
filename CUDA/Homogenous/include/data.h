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
#define MAX_NEIGHBORS 20
// #define EDGE_PATH "../../Data/Homogenous/google/webGoogle.csv"
// #define CLUSTER_PATH "../../Data/Homogenous/google/Cluster_Assignment.csv"

#define EDGE_PATH "../../Data/Homogenous/rand/rand_net.csv"
#define GRAPH_DATA_PATH "../../Data/Homogenous/rand/rand_net_info.csv"
#define CLUSTER_PATH "../../Data/Homogenous/google/Cluster_Assignment.csv"
#define PTR_PATH "../../Data/Homogenous/google/ptr_Assignment.csv"
#define LIST_PATH "../../Data/Homogenous/google/list_check.csv"



struct edge{
    unsigned int end, start;
    unsigned int cluster;
};

struct vert_hash_table{
    unsigned int local_vert;
    unsigned int global_vert;
};

__host__ void Check_Out_csv_edge(edge* edge_list, int size);

__host__ void return_edge_list(string path, edge* arr);

__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size);

__host__ void CSR_Graph(string path, unsigned int node_size, unsigned int edge_size, unsigned int* src_ptr, unsigned int* succ, unsigned int* deg_arr);

__host__ void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);

__host__ void Check_Out_pref_sum(unsigned int* list_1, unsigned int* list_2, int size);

__host__ int getMax_cluster(edge* edge_list, int n);

__host__ void cpu_countSort(edge* arr, int n, int exp);

__host__ void cpu_radixsort(edge* arr, int n);

__host__ void Check_Out_ptr(unsigned int* edge_list, int size);

__global__ void bit_exclusive_scan(unsigned int* bits, unsigned int* bits_2, unsigned int* bits_3, unsigned int size);

__global__ void Sort_Cluster(edge* edgelist, unsigned int* table, unsigned int size,unsigned int iter);

__host__ void Org_Vertex_Helper(edge* h_edge, unsigned int* h_src_ptr, unsigned int* h_succ, unsigned int* h_deg, unsigned int size, unsigned int node_size);

__global__ void Swap(edge* edge_list, edge* edge_list_2, unsigned int* table, unsigned int* table_2, long int size, unsigned int iter);

__global__ void Random_Edge_Placement(edge *edges, double rand_num);

__global__ void Degree_Based_Placement(edge* edges, unsigned int* deg_arr, double rand_num, unsigned int size);

__global__ void fin_exclusive_scan(unsigned int* bits_3, unsigned int size);

__global__ void copy_edge_list(edge* edge_1, edge* edge_2, unsigned int size);

__global__ void Histogram_1(edge* edgelist, unsigned int* hist_bin, unsigned int size);

__global__ void Kogge_Stone_Hist_Reduct(unsigned int* hist_bin, unsigned int* fin_bin, int size);

__global__ void Hist_Prefix_Sum(unsigned int* fin_bin, unsigned int* fin_bin_2);

__global__ void final_scan_commit(unsigned int* bits_2, unsigned int* bits_3, unsigned int size);

__global__ void First_Init(float* rand_frog, unsigned int* d_frog, unsigned int node_size, unsigned int edge_size);

__global__ void FrogWild(edge* edgelist, unsigned int* d_src, unsigned int* d_succ,
unsigned int* d_c, unsigned int* d_frogs, unsigned int* edge_ptr,unsigned int* ctr_ptr,
unsigned int size, unsigned int iter);

__global__ void fin_acc(unsigned int* table, unsigned int k, float* acc);

__global__ void acc_accum(unsigned int* approx, unsigned int* pagerank, unsigned int* table, unsigned int k);

__global__ void gen_backward_start_mask(edge* edgelist, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* start_mask, unsigned int size);

__global__ void gen_backward_end_mask(edge* edgelist, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* end_mask, unsigned int size);

__global__ void scan_start_mask(unsigned int* start_mask, unsigned* compct_start, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size);

__global__ void Prefix_Scan_Cmpt(unsigned int* mask, unsigned int* cmpt, unsigned int size, unsigned int block);

__global__ void scan_end_mask(unsigned int* end_mask, unsigned* compct_end, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size);

__global__ void Scanned_To_Compact(unsigned int* cmpt, unsigned int* scanned, unsigned int* new_size, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size);

__global__ void Final_Compression(unsigned int* cmpt, unsigned int* new_size, edge* edge_list, unsigned int* new_idx, unsigned int* out, int type);

__global__ void Find_Max_Cluster(unsigned int* ctr_table, unsigned int* max_val);

__global__ void unq_exclusive_scan(unsigned int* len, unsigned int* unq_ptr);