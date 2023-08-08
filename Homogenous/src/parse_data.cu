#include "../include/data.h"
#include "../include/GPUErrors.h"

#define BLOCKS 16
#define TPB 256


__host__ void return_list(string path, int** arr){
    ifstream data;
    data.open(path);
    string line,word;
    int count=0;
    int column=0;
    cout<<data.is_open()<<endl;
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); //Stream Class to operate on strings
            while(getline(stream_data,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    arr[count-1][column]=stoi(word);
                    column++;
                }
                //Extract data until ',' is found
            }
            column=0;
            count++;
        }
    }
    else{
        cout<<"Cannot open file"<<endl;
    }
    data.close();
}


__host__ void split_list(int** arr, int* subarr_1, int* subarr_2, int size){
    for(int i=0; i<size;i++){
        subarr_1[i]=arr[i][0];
        subarr_2[i]=arr[i][1];
    }
}


__global__ void Sort_Cluster(int* cluster, int* vertex, int* table, int size,int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    int tid= threadIdx.x;
    __shared__ int shared_cluster[TPB];
    __shared__ int shared_vertex[TPB];
    __shared__ int bits[TPB];
    __shared__ int ex_bits[TPB];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_cluster[tid]=cluster[idx];
        shared_vertex[tid]=vertex[idx];
    }
    __syncthreads();

    //Perform sorting
    unsigned int key, bit, vert_val;
    if(idx<size){
        key=shared_cluster[tid];
        vert_val=shared_vertex[tid];
        bit=(key>>iter) & 1;
        bits[tid]=bit;
    }
    __syncthreads();
    //Perform exclusive scan
    if(tid<TPB && tid!=0){
        ex_bits[tid]=bits[tid-1];
    }
    else{
        ex_bits[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        int temp;
        if(tid>=stride){
            temp=ex_bits[tid]+ex_bits[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            ex_bits[tid]=temp;
        }
    }
    if(tid<TPB){
        bits[tid]=ex_bits[tid];
    }
    __syncthreads();
    if(idx<size){
        int num_one_bef=bits[tid];
        int num_one_total=bits[TPB-1];
        int dst = (bit==0)? (tid - num_one_bef):(TPB-num_one_total-num_one_bef);
        shared_vertex[dst]=vert_val;
        shared_cluster[dst]=key;
    }
    __syncthreads();
    if(tid==0){
        table[blockIdx.x]=blockDim.x-bits[blockDim.x-1];
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=bits[blockDim.x-1];
    }
    __syncthreads();
    if(idx<size){
        vertex[idx]=shared_vertex[tid];
        cluster[idx]=shared_cluster[tid];
    }
    __syncthreads();
    if(idx==0){
        //Have thread 0 launch the kernel to perform the sum
        //Save the number of 0's
        bit_exclusive_scan<<<1,2*gridDim.x,0,cudaStreamTailLaunch>>>(table,2*gridDim.x);
    }
    __syncthreads();
    // // //We now have the pointer values in global memory to store data
    if(idx<size){
        if(tid<=blockDim.x-bits[blockDim.x-1]){
            cluster[table[blockIdx.x]+tid]=shared_cluster[tid];
            vertex[table[blockIdx.x]+tid]=shared_vertex[tid];
        }
        else{
            cluster[table[blockIdx.x+gridDim.x]+tid]=shared_cluster[tid];
            vertex[table[blockIdx.x+gridDim.x]+tid]=shared_vertex[tid];
        }
    }
    __syncthreads();
}

__global__ void Swap(int* cluster, int* vertex, int* table, int* table_2,  int size){
    int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    int tid= threadIdx.x;
    // const int cluster_size= size/gridDim.x+1;
    __shared__ int shared_cluster[TPB];
    __shared__ int shared_vertex[TPB];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_cluster[tid]=cluster[idx];
        shared_vertex[tid]=vertex[idx];
    }
    __syncthreads();   
    if(idx<size){
        if(tid<=table_2[blockIdx.x]){
            cluster[table[blockIdx.x]+tid]=shared_cluster[tid];
            vertex[table[blockIdx.x]+tid]=shared_vertex[tid];
        }
        else{
            cluster[table[blockIdx.x+gridDim.x]+tid]=shared_cluster[tid];
            vertex[table[blockIdx.x+gridDim.x]+tid]=shared_vertex[tid];
        }
    }
    __syncthreads();
}

__global__ void bit_exclusive_scan(int* bits, int* bit_2,int size){
    int tid=threadIdx.x;
    __shared__ int ex_bits[TPB];
    if(tid<size && tid!=0){
        ex_bits[tid]=bits[tid-1];
    }
    else{
        ex_bits[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        int temp;
        if(tid>=stride){
            temp=ex_bits[tid]+ex_bits[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            ex_bits[tid]=temp;
        }
    }
    if(tid<TPB){
        bit_2[tid]=ex_bits[tid];
    }
    __syncthreads();
}



__host__ void Org_Vertex_Helper(int* h_cluster, int* h_vertex, int size){
    //Allocate memory for vertex and cluster info
    int* d_vertex;
    int* d_cluster;
    int* d_table;
    int* d_table_2;

    int threads_per_block=TPB;
    int blocks_per_grid= size/threads_per_block+1;

    if(!HandleCUDAError(cudaMalloc((void**) &d_vertex, size*sizeof(int)))){
        cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_cluster,size*sizeof(int)))){
        cout<<"Unable to allocate memory for the cluster data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_table,(2*blocks_per_grid)*sizeof(int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table,0,(2*blocks_per_grid)*sizeof(int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMalloc((void**) &d_table_2,(2*blocks_per_grid)*sizeof(int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table_2,0,(2*blocks_per_grid)*sizeof(int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMemcpy(d_vertex,h_vertex,size*sizeof(int), cudaMemcpyHostToDevice))){
        cout<<"Unable to copy vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_cluster,h_cluster,size*sizeof(int), cudaMemcpyHostToDevice))){
        cout<<"Unable to copy cluster data"<<endl;
    }

    for(int i=0; i<32;i++){
        Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_cluster,d_vertex,d_table,size,i);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Sort Cluster"<<endl;
        }
        // bit_exclusive_scan<<<1,2*blocks_per_grid>>>(d_table,d_table_2,2*blocks_per_grid);
        // if(!HandleCUDAError(cudaDeviceSynchronize())){
        //     cout<<"Unable to synchronize with host exclusive scan"<<endl;
        // }
        // Swap<<<blocks_per_grid,threads_per_block>>>(d_cluster,d_vertex,d_table_2,d_table,size);
        // if(!HandleCUDAError(cudaDeviceSynchronize())){
        //     cout<<"Unable to synchronize with host swap"<<endl;
        // }
    }

    if(!HandleCUDAError(cudaMemcpy(h_vertex,d_vertex,size*sizeof(int),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_cluster,d_cluster,size*sizeof(int),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back cluster data"<<endl;
    }
    HandleCUDAError(cudaFree(d_cluster));
    HandleCUDAError(cudaFree(d_vertex));
    HandleCUDAError(cudaFree(d_table));
    HandleCUDAError(cudaDeviceReset());   
}



