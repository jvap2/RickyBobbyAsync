#include "../include/data.h"
#include "../include/GPUErrors.h"


/*We are going to simulate the same function as the nvgraph page rank
it utilizes cublas and cusparse*/


__global__ void Gen_P(float* weight_P,unsigned int* src, unsigned int* succ, unsigned int node_size){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < node_size && y < node_size){
        weight_P[y*node_size + x] = 1.0f/(src[x+1]-src[x]);
    }
}
