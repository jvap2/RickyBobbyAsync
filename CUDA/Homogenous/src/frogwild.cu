#include "../include/data.h"
#include "../include/GPUErrors.h"

#define TPB 256

__global__ void acc_accum(unsigned int* approx, unsigned int* pagerank, unsigned int* table, unsigned int k){
    unsigned int idx=threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_table[TPB];
    if(idx<k){
        if(approx[idx]==pagerank[idx]){
            local_table[tid]=1;
        }
        else{
            local_table[tid]=0;
        }
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=local_table[tid]+local_table[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            local_table[tid]=temp;
        }
    }
    if(tid==(blockDim.x-1)){
        local_table[tid]=table[idx];
    }
}

__global__ float fin_acc(unsigned int* table, unsigned int k){
    float acc;
    unsigned int tid = threadIdx.x;
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=table[tid]+table[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            table[tid]=temp;
        }
    }
    if(tid==(blockDim.x-1)){
        acc=float(table[tid])/float(k);
    }
    return acc;
}

