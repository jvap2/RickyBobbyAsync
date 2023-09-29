#include "../include/data.h"
#include "../include/GPUErrors.h"


/*We are going to simulate the same function as the nvgraph page rank
it utilizes cublas and cusparse*/


__global__ void Init_P(float* P, unsigned int node_size, float* damp){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < node_size && y < node_size){
            P[y*node_size + x] = (*damp/(1.0*node_size));
    }
}

__global__ void Gen_P(float* weight_P,edge* edgelist, unsigned int* src, unsigned int node_size, float* damp){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < node_size){
        unsigned int start = edgelist[x].start;
        unsigned int end = edgelist[x].end;
        unsigned int out = src[start+1]-src[start];
        if(out!=0){
            weight_P[end*node_size+start]+=(1.0-*damp)/(out*1.0);
        }
    }

}


__global__ void Gen_P_Mem_eff(float* weight_P, unsigned int* src, unsigned int* succ, unsigned int node_size, float* damp){
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<node_size){
        //We need to find a node in the src
        //We will then iterate through the succ, and access the src[succ+1] to src[succ] to 
        // get the out degree
        unsigned int num_succ=src[idx+1]-src[idx];
        if(num_succ!=0){
            for(unsigned int i=src[idx]; i<src[idx+1]; i++){
                unsigned int succ_node = succ[i];//Get the node number of the successor
                weight_P[succ_node*node_size+idx]+=(1.0-*damp)/(1.0f*num_succ);
            }
        }
    }
}

__global__ void Init_Pr(float* pr_vector, unsigned int node_size){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(x < node_size){
        pr_vector[x] = 1.0f/node_size;
    }
}

__host__ void PageRank(float* pr_vector, unsigned int* h_indices, unsigned int* global_src, unsigned int* global_succ, float damp, unsigned int node_size, unsigned int edge_size, unsigned int max_iter, float tol){
    float alpha = 1.0; 
    float beta = 0.0;
    float tol_temp=100.0f;
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
    unsigned int tpb_2d = 16;
    unsigned int blocks_2d = (node_size+tpb_2d-1)/tpb_2d;
    dim3 Threads(tpb_2d, tpb_2d);
    dim3 Blocks(blocks_2d, blocks_2d);
    unsigned int blocks_edge = (edge_size+tpb-1)/tpb;
    edge* d_edgelist;
    if(!HandleCUDAError(cudaMalloc((void**)&d_edgelist, edge_size*sizeof(edge)))){
        cout<<"Error allocating memory for edgelist"<<endl;
    }
    // edge* h_edgelist = (edge*)malloc(sizeof(edge)*edge_size);
    // return_edge_list(EDGE_PATH,h_edgelist);
    // if(!HandleCUDAError(cudaMemcpy(d_edgelist, h_edgelist, edge_size*sizeof(edge), cudaMemcpyHostToDevice))){
    //     cout<<"Error copying edgelist to device"<<endl;
    // }
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
    cout<< "2D grid dim: "<<Blocks.x<<" "<<Blocks.y<<endl;
    cout<< "2D block dim: "<<Threads.x<<" "<<Threads.y<<endl;
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
                static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }


    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("  Total amount of constant memory:               %zu bytes\n",
            deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
            deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
            deviceProp.sharedMemPerMultiprocessor);
    }


    size_t free_byte ;
    size_t total_byte ;
    if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
        cout<<"Error getting memory info"<<endl;
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

    Init_P<<<Blocks,Threads>>>(d_P, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device with Initializing P"<<endl;
    }
    // Gen_P<<<blocks_edge,tpb>>>(d_P, d_edgelist, d_global_src, edge_size, d_damp);
    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Error synchronizing device with Generating P"<<endl;
    // }
    // if(!HandleCUDAError(cudaFree(d_edgelist))){
    //     cout<<"Error freeing memory for edgelist"<<endl;
    // }
    // free(h_edgelist);
    Gen_P_Mem_eff<<<blocks,tpb>>>(d_P, d_global_src, d_global_succ, node_size, d_damp);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device with Generating P"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_global_src))){
        cout<<"Error freeing memory for global_src"<<endl;
    }
    
    //Now, we need to initialize the pr_vector
    if(!HandleCUDAError(cudaMalloc((void**)&d_pr_vector, node_size*sizeof(float)))){
        cout<<"Error allocating memory for pr_vector"<<endl;
    }
    Init_Pr<<<blocks,tpb>>>(d_pr_vector, node_size);
    if(!HandleCUDAError(cudaMalloc((void**)&dr_pr_vector_temp, node_size*sizeof(float)))){
        cout<<"Error allocating memory for dr_pr_vector_temp"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(dr_pr_vector_temp, 0, node_size*sizeof(float)))){
        cout<<"Error setting dr_pr_vector_temp to 0"<<endl;
    }
    cublasHandle_t handle;
    if(!HandleCUBLASError(cublasCreate(&handle))){
        cout<<"Error creating cublas handle"<<endl;
    }
    cout<<"Performing PageRank"<<endl;
    unsigned int iter_temp=max_iter;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    while(max_iter>0 && tol_temp>tol){
        cublasSgemv_v2(handle, CUBLAS_OP_T, node_size, node_size, &alpha, d_P, node_size, d_pr_vector, 1, &beta, dr_pr_vector_temp, 1);
        cublasSnrm2_v2(handle, node_size, dr_pr_vector_temp, 1, &norm_temp);
        cublasSnrm2_v2(handle, node_size, d_pr_vector, 1, &norm);
        tol_temp = fabsf(norm_temp-norm);
        cublasScopy_v2(handle, node_size, dr_pr_vector_temp, 1, d_pr_vector, 1);
        max_iter--;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout<<"Time elapsed PageRank: "<<milliseconds<<" ms"<<endl;
    cout<<"Converged in "<<iter_temp-max_iter<<" iterations"<<endl;
    cout<<"Tolerance: "<<tol_temp<<endl;
    unsigned int *d_indices;

    if(!HandleCUDAError(cudaMalloc((void**)&d_indices, node_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_indices"<<endl;
    }
    thrust::sequence(thrust::device, d_indices, d_indices+node_size);
    thrust::stable_sort_by_key(thrust::device, d_pr_vector, d_pr_vector+node_size, d_indices, thrust::greater<float>());
    if(!HandleCUDAError(cudaMemcpy(h_indices, d_indices, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
        cout<<"Error copying d_indices to host"<<endl;
    }
    cout<<"PageRank finished"<<endl;
    if(!HandleCUDAError(cudaMemcpy(pr_vector, d_pr_vector, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
        cout<<"Error copying pr_vector to host"<<endl;
    }
    if(!HandleCUDAError(cudaFree(d_P))){
        cout<<"Error freeing memory for P"<<endl;
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



__host__ void Export_pr_vector(float* pr_vector, unsigned int* indices, unsigned int node_size){
    ofstream myfile;
    myfile.open(CUBLAS_PR_PATH);
    myfile<<"Node, PageRank"<<endl;
    for(unsigned int i=0; i<node_size; i++){
        myfile<<indices[i]<<", "<<pr_vector[i]<<endl;
    }
    myfile.close();
}


__host__ void Print_Matrix(float* matrix, unsigned int node_size){
    int non_zero_count=0;
    for(unsigned int i=0; i<node_size; i++){
        for(unsigned int j=0; j<node_size; j++){
            if(matrix[i*node_size+j]>0){
                non_zero_count++;
            }
            cout<<matrix[i*node_size+j]<<" ";
        }
        cout<<endl;
    }
    printf("Non zero count: %d\n", non_zero_count);
}