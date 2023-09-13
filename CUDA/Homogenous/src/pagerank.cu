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

__host__ void PageRank(unsigned int* P, unsigned int* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size){
    cout<<"Performing PageRank"<<endl;
    float* d_P;
    unsigned int* d_global_src, *d_global_succ;
    if(!HandleCUDAError(cudaMalloc((void**)&d_P, (node_size*node_size)*sizeof(float)))){
        cout<<"Error allocating memory for d_P"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_global_src"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, (edge_size)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_global_succ"<<endl;
    }
    Gen_P<<<BLOCKS,TPB>>>(d_P, d_global_src, d_global_succ, node_size);
    float* d_p_vals;
    unsigned int *d_p_src, *d_p_succ;
    void* dBuffer = NULL;
    size_t bufferSize = 0;
    if(!HandleCUDAError(cudaMalloc((void**)&d_p_src, (node_size+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_p_src"<<endl;
    }
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matP_sparse;
    cusparseDnMatDescr_t matP_full;
    cusparseDnVecDescr_t vec_pr;
    if(!HandleCUSparseError(cusparseCreate(&handle))){
        cout<<"Error creating handle"<<endl;
    }
    if(!HandleCUSparseError(cusparseCreateDnMat(&matP_full, node_size, node_size, node_size, d_P, CUDA_R_32F, CUSPARSE_ORDER_ROW))){
        cout<<"Error creating full matrix"<<endl;
    }
    if(!HandleCUSparseError(cusparseCreateCsr(&matP_sparse, node_size, node_size, 0,d_p_src, NULL, NULL,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))){
        cout<<"Error creating sparse matrix"<<endl;
    }
    if(!HandleCUSparseError(cusparseDenseToSparse_bufferSize(handle, matP_full, matP_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize))){
        cout<<"Error getting buffer size"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dBuffer, bufferSize))){
        cout<<"Error allocating memory for dBuffer"<<endl;
    }
    if(!HandleCUSparseError(cusparseCreateDnVec(&vec_pr, node_size, pr_vector, CUDA_R_32F))){
        cout<<"Error creating vector"<<endl;
    }

    int64_t num_r_tmp, num_n_tmp, num_nnz_tmp;

    if(!HandleCUSparseError(cusparseSpMatGetSize(matP_sparse, &num_r_tmp, &num_n_tmp, &num_nnz_tmp))){
        cout<<"Error performing analysis"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_p_vals, (num_nnz_tmp)*sizeof(float)))){
        cout<<"Error allocating memory for d_p_vals"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_p_succ, (num_nnz_tmp)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_p_succ"<<endl;
    }

    if(!HandleCUSparseError(cusparseCsrSetPointers(matP_sparse, d_p_src, d_p_succ, d_p_vals))){
        cout<<"Error setting pointers"<<endl;
    }
    if(!HandleCUSparseError(cusparseDenseToSparse_convert(handle, matP_full, matP_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer))){
        cout<<"Error converting dense to sparse"<<endl;
    }
    if(!HandleCUSparseError(cusparseDestroyDnMat(matP_full))){
        cout<<"Error destroying full matrix"<<endl;
    }

    if(!HandleCUSparseError(cusparseDestroySpMat(matP_sparse))){
        cout<<"Error destroying sparse matrix"<<endl;
    }
    if(!HandleCUSparseError(cusparseDestroy(handle))){
        cout<<"Error destroying handle"<<endl;
    }
}

