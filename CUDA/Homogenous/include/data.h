#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>   
#include<algorithm>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../include/GPUErrors.h"
//Google
#define BLOCKS 12
#if BLOCKS>=48
#define TPB 128
#else
#define TPB 256
#endif
// #define EDGES 5105039
// #define NODES 875713
#define MAX_NEIGHBORS 20
// #define EDGE_PATH "../../Data/Homogenous/google/webGoogle.csv"
// #define CLUSTER_PATH "../../Data/Homogenous/google/Cluster_Assignment.csv"

#define EDGE_PATH "../../Data/Homogenous/rand/rand_net.csv"
#define GRAPH_DATA_PATH "../../Data/Homogenous/rand/rand_net_info.csv"
#define CLUSTER_PATH "../../Data/Homogenous/rand/Cluster_Assignment_Norm.csv"
#define PTR_PATH "../../Data/Homogenous/google/ptr_Assignment.csv"
#define LIST_PATH "../../Data/Homogenous/google/list_check.csv"
#define REPLICA_PATH "../../Data/Homogenous/replica_counts.csv"
#define POWER_GRAPH_EDGE_PATH "../../Data/Homogenous/rand/power_rand_net.csv"
#define POWER_GRAPH_CLUSTER_PATH "../../Data/Homogenous/rand/power_Cluster_Assignment.csv"
#define POWER_GRAPH_DATA_PATH "../../Data/Homogenous/rand/power_rand_net_info.csv"
#define POWER_REPLICA_PATH "../../Data/Homogenous/rep_power_counts.csv"
#define PTR_CTR_PATH "../../Data/Homogenous/rand/ptr_ctr_Assignment.csv"
#define UNQ_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Unique_C.csv"
#define HIST_PATH "../../Data/Homogenous/rand/hist_Assignment.csv"
#define LOCAL_SRC_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Source_C.csv"
#define LOCAL_SUCC_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Successor_C.csv"
#define SRC_CTR_PTR_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Source_Ptr_Ctr_C.csv"
#define UNQ_CTR_PTR_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Unique_Ptr_Ctr_C.csv"
#define H_CTR_PTR_PATH "../../Data/Homogenous/rand/check/Local_Cluster_Hist_Ptr_Ctr_C.csv"
#define DEG_PATH "../../Data/Homogenous/rand/check/Node_Degree_C.csv"
#define REPLICA_STAT_PATH "../../Data/Homogenous/rand/check/Replica_Stat_C.csv"

struct edge{
    unsigned int end, start;
    unsigned int cluster;
};

struct vert_hash_table{
    unsigned int local_vert;
    unsigned int global_vert;
};

struct replica_tracker{
    unsigned int clusters[BLOCKS];
    unsigned int num_replicas;
};

/*CPU FUNCTIONS*/

__host__ void Check_Out_csv_edge(edge* edge_list, int size);

__host__ void return_edge_list(string path, edge* arr);

__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size);

__host__ void CSR_Graph(string path, unsigned int node_size, unsigned int edge_size, unsigned int* src_ptr, unsigned int* succ, unsigned int* deg_arr);

__host__ void get_graph_info(string path, unsigned int* nodes, unsigned int* edges);

__host__ void Check_Out_pref_sum(unsigned int* list_1, unsigned int* list_2, int size);

__host__ void check_out_replicas(string path,unsigned int* replicas, unsigned int node_size);

__host__ int getMax_cluster(edge* edge_list, int n);

__host__ void cpu_countSort(edge* arr, int n, int exp);

__host__ void cpu_radixsort(edge* arr, int n);

__host__ void Check_Out_ptr(unsigned int* edge_list, int size);

__host__ void Capture_Node_Degree(edge* edge_list, unsigned int* deg_arr, unsigned int size);

__host__ void Check_Out_Unq(unsigned int* h_unq, int size);

__host__ void Check_Out_Ptr_Ctr(unsigned int* h_ctr, unsigned int* h_ptr, int size);

__host__ void Check_Repeats(edge* edge_list, unsigned int size);

__host__ void Gen_Local_Src(edge* edge_list, unsigned int* src_ptr,unsigned int* temp_src, unsigned int* unq, unsigned int* h_unq_ctr, unsigned int* h_unq_ptr,
unsigned int* h_ctr, unsigned int* h_ptr);

__host__ void Generate_Renum_Edgelists(edge* edge_list, edge* edge_list_2, unsigned int* unq, unsigned int* h_ptr, unsigned int* h_ctr, unsigned int* h_unq_ctr, unsigned int* h_unq_ptr);

__host__ void Generate_Local_Succ(edge* edgelist, unsigned int* local_src, unsigned int* local_succ, unsigned int* h_unq_ctr, unsigned int* h_unq_ptr, unsigned int* h_ptr);

__host__ void Export_Local_Succ(unsigned int* local_succ, unsigned int* h_ptr, unsigned int* h_ctr);

__host__ void Export_Local_Src(unsigned int* local_src, unsigned int* h_ptr, unsigned int* h_ctr);

__host__ void Export_Unq(unsigned int* unq, unsigned int* h_unq_ptr, unsigned int* h_unq_ctr);

__host__ void Export_Src_Ctr_Ptr(unsigned int* src_ptr, unsigned int* src_ctr);

__host__ void Export_Unq_Ctr_Ptr(unsigned int* h_unq_ptr, unsigned int* h_unq_ctr);

__host__ void Export_H_Ctr_Ptr(unsigned int* h_ptr, unsigned int* h_ctr);

__host__ void Export_Degree(unsigned int* deg, unsigned int node_size);

__host__ void Export_Replica_Stats(replica_tracker* h_replica, unsigned int node_size);

__host__ void Import_Local_Src(unsigned int* local_src);

__host__ void Import_Local_Succ(unsigned int* local_succ);

__host__ void Import_Unique(unsigned int* unq);

__host__ void Import_H_Ctr_Ptr(unsigned int* h_ctr, unsigned int* h_ptr);

__host__ void Import_Src_Ctr_Ptr(unsigned int* src_ctr, unsigned int* src_ptr);

__host__ void Import_Unq_Ptr_Ctr(unsigned int* unq_ptr, unsigned int* unq_ctr);

__host__ void Import_Degree(unsigned int* deg, unsigned int node_size);

__host__ void Import_Replica_Stats(replica_tracker* h_replica, unsigned int node_size);
/*HELPER FUNCTION AND KERNELS*/

__host__ void FrogWild(unsigned int* local_succ, unsigned int* local_src, unsigned int* unq, unsigned int* c, unsigned int* k, unsigned int* src_ptr, 
unsigned int* unq_ptr, unsigned int* h_ptr, unsigned int* degree, replica_tracker* h_replica, int node_size, unsigned int edge_size);

__global__ void bit_exclusive_scan(unsigned int* bits, unsigned int* bits_2, unsigned int* bits_3, unsigned int size);

__global__ void Sort_Cluster(edge* edgelist, unsigned int* table, unsigned int size,unsigned int iter);

__host__ void Org_Vertex_Helper(edge* h_edge, unsigned int* replica_count, replica_tracker* h_tracker, unsigned int* h_deg, unsigned int* h_ctr, unsigned int* h_ptr,unsigned int size, unsigned int node_size);

__global__ void Swap(edge* edge_list, edge* edge_list_2, unsigned int* table, unsigned int* table_2, long int size, unsigned int iter);

__global__ void Random_Edge_Placement(edge *edges, double rand_num, unsigned int size);

__global__ void Degree_Based_Placement(edge* edges, unsigned int* deg_arr, double rand_num, replica_tracker* d_rep, unsigned int size);

__global__ void Finalize_Replica_Tracker(replica_tracker* d_rep, unsigned int node_size);

__global__ void fin_exclusive_scan(unsigned int* bits_3, unsigned int size);

__global__ void final_scan_commit_scan(unsigned int* list, unsigned int* end_vals, unsigned int ptr, unsigned int size);

__global__ void copy_edge_list(edge* edge_1, edge* edge_2, unsigned int size);

__global__ void Histogram_1(edge* edgelist, unsigned int* hist_bin, unsigned int size);

__global__ void Kogge_Stone_Hist_Reduct(unsigned int* hist_bin, unsigned int* fin_bin, int size);

__global__ void Hist_Prefix_Sum(unsigned int* fin_bin, unsigned int* fin_bin_2);

__global__ void final_scan_commit(unsigned int* bits_2, unsigned int* bits_3, unsigned int size);

__global__ void First_Init(float* rand_frog, unsigned int* K, unsigned int node_size, unsigned int sublinear_size);

__global__ void fin_acc(unsigned int* table, unsigned int k, float* acc);

__global__ void acc_accum(unsigned int* approx, unsigned int* pagerank, unsigned int* table, unsigned int k);

__global__ void gen_backward_mask(unsigned int* global_list, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* start_mask, unsigned int size);

__global__ void scan_mask(unsigned int* start_mask, unsigned* compct_start, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size);

__global__ void Prefix_Scan_Cmpt(unsigned int* mask, unsigned int* cmpt, unsigned int size);

__global__ void Scanned_To_Compact(unsigned int* cmpt, unsigned int* scanned, unsigned int* new_size, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size);

__global__ void Final_Compression(unsigned int* cmpt, unsigned int* new_size, unsigned int* in, unsigned int* new_idx, unsigned int* out);

__global__ void Find_Max_Cluster(unsigned int* ctr_table, unsigned int* max_val);

__global__ void unq_exclusive_scan(unsigned int* len, unsigned int* unq_ptr);

// __global__ void Total_Unq_Ptr(unsigned int* start_ptr, unsigned int* end_ptr, unsigned int* fin_ptr);

__global__ void temp_Copy_Start_End(edge* edge_list, unsigned int* start, unsigned int* end, unsigned int edge_size);

__global__ void Naive_Merge_Sort(unsigned int* start, unsigned int* end, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* unq);

__global__ void Apply(unsigned int* unq, unsigned int* unq_ptr, unsigned int* K, unsigned int* C, unsigned int* num_loc_K, unsigned int* local_K_idx, unsigned int iter, float* p_t, curandState* d_state);

__global__ void Gather(unsigned int* K, unsigned int* C, unsigned int* unq, unsigned int* unq_ptr, unsigned int* num_local_C, unsigned int* num_local_K,
unsigned int* local_K, unsigned int* local_C, unsigned int* local_K_idx, unsigned int* local_C_idx);
/*DEVICE FUNCTIONS*/

__device__ unsigned int co_rank(unsigned int* start, unsigned int* end, int m, int n, int k);

__device__ __host__ void merge_sequential(unsigned int* start, unsigned int* end, int m, int n, unsigned int* unq);