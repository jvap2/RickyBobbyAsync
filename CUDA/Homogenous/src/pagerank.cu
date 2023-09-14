#include "../include/data.h"
#include "../include/GPUErrors.h"


/*We are going to simulate the same function as the nvgraph page rank
it utilizes cublas and cusparse*/


__global__ void Init_P(float* P, unsigned int node_size, float* damp){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < node_size && y < node_size){
            P[y*node_size + x] = *damp/(1.0f*node_size);
    }
}

__global__ void Gen_P(float* weight_P,edge* edgelist, unsigned int* src, unsigned int node_size, float* damp){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < node_size){
        unsigned int start =edgelist[x].start;
        unsigned int end = edgelist[x].end;
        unsigned int out = src[start+1]-src[start];
        weight_P[end*node_size+start]+=(1.0f-*damp)/out;
    }

}

__global__ void Init_Pr(float* pr_vector, unsigned int node_size){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < node_size){
        pr_vector[x] = 1.0f/node_size;
    }
}

__host__ void PageRank(float* pr_vector, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol){
    float alpha = 1.0f; 
    float beta = 0.0f;
    float tol_temp=100.f;
    float* d_P;
    unsigned int* d_global_src;
    unsigned int* d_global_succ;
    float* d_pr_vector;
    float* dr_pr_vector_temp;
    float* d_damp;
    float norm=0;
    float norm_temp=0;
    unsigned int tpb = 256;
    unsigned int blocks = (node_size+tpb-1)/tpb;
    dim3 Threads(tpb, tpb);
    dim3 Blocks(blocks, blocks);
    unsigned int blocks_edge = (edge_size+tpb-1)/tpb;
    edge* d_edgelist;
    if(!HandleCUDAError(cudaMalloc((void**)&d_edgelist, edge_size*sizeof(edge)))){
        cout<<"Error allocating memory for edgelist"<<endl;
    }
    edge* h_edgelist = new edge[edge_size];
    return_edge_list(EDGE_PATH,h_edgelist);;
    for(int i=0; i<edge_size; i++){
        cout<<h_edgelist[i].start<<" "<<h_edgelist[i].end<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_edgelist, h_edgelist, edge_size*sizeof(edge), cudaMemcpyHostToDevice))){
        cout<<"Error copying edgelist to device"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_P, node_size*node_size*sizeof(float)))){
        cout<<"Error allocating memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for global_src"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, edge_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_src to device"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, edge_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying global_succ to device"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_damp, sizeof(float)))){
        cout<<"Error allocating memory for damp"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_damp, &damp, sizeof(float), cudaMemcpyHostToDevice))){
        cout<<"Error copying damp to device"<<endl;
    }

    Init_P<<<Blocks,Threads>>>(d_P, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    Gen_P<<<blocks_edge,tpb>>>(d_P, d_edgelist, d_global_src, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    //Now, we need to initialize the pr_vector
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr_vector, node_size*sizeof(float)))){
        cout<<"Error allocating memory for pr_vector"<<endl;
    }
    Init_Pr<<<blocks,tpb>>>(d_pr_vector, node_size);
    if(!HandleCUDAError(cudaMalloc((void**)&dr_pr_vector_temp, node_size*sizeof(float)))){
        cout<<"Error allocating memory for dr_pr_vector_temp"<<endl;
    }
    Init_Pr<<<blocks,tpb>>>(dr_pr_vector_temp, node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    cublasHandle_t handle;
    if(!HandleCUBLASError(cublasCreate(&handle))){
        cout<<"Error creating cublas handle"<<endl;
    }
    cout<<"Performing PageRank"<<endl;
    unsigned int iter_temp=max_iter;
    while(max_iter>0 && tol_temp>tol){
        cublasSgemv_v2(handle, CUBLAS_OP_N, node_size, node_size, &alpha, d_P, node_size, d_pr_vector, 1, &beta, dr_pr_vector_temp, 1);
        cublasSasum_v2(handle, node_size, dr_pr_vector_temp, 1, &norm_temp);
        cublasSasum_v2(handle, node_size, d_pr_vector, 1, &norm);
        printf("Norm_temp: %f\n", norm_temp);
        printf("Norm: %f\n", norm);
        tol_temp = fabsf(norm_temp-norm);
        norm_temp=0;
        norm=0;
        cublasScopy_v2(handle, node_size, dr_pr_vector_temp, 1, d_pr_vector, 1);
        max_iter--;
    }
    cout<<"Converged in "<<iter_temp-max_iter<<" iterations"<<endl;
    cout<<"Tolerance: "<<tol_temp<<endl;

    cout<<"PageRank finished"<<endl;
    float* P_test = new float[node_size*node_size]{0};
    if(!HandleCUDAError(cudaMemcpy(P_test, d_P, node_size*node_size*sizeof(float), cudaMemcpyDeviceToHost))){
        cout<<"Error copying P to host"<<endl;
    }
    Print_Matrix(P_test, node_size);
    if(!HandleCUDAError(cudaMemcpy(pr_vector, d_pr_vector, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
        cout<<"Error copying pr_vector to host"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_P))){
        cout<<"Error freeing memory for P"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_global_src))){
        cout<<"Error freeing memory for global_src"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_global_succ))){
        cout<<"Error freeing memory for global_succ"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_pr_vector))){
        cout<<"Error freeing memory for pr_vector"<<endl;
    }
    if(!HandleCUDAError(cudaFree(dr_pr_vector_temp))){
        cout<<"Error freeing memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_damp))){
        cout<<"Error freeing memory for damp"<<endl;
    }
    if(!HandleCUBLASError(cublasDestroy(handle))){
        cout<<"Error destroying cublas handle"<<endl;
    }
}



__host__ void export_pr_vector(float* pr_vector, unsigned int node_size){
    ofstream myfile;
    myfile.open(CUBLAS_PR_PATH);
    myfile<<"Node, PageRank"<<endl;
    for(unsigned int i=0; i<node_size; i++){
        myfile<<i<<", "<<pr_vector[i]<<endl;
    }
    myfile.close();
}


__host__ void Print_Matrix(float* matrix, unsigned int node_size){
    for(unsigned int i=0; i<node_size; i++){
        for(unsigned int j=0; j<node_size; j++){
            cout<<matrix[i*node_size+j]<<" ";
        }
        cout<<endl;
    }
}