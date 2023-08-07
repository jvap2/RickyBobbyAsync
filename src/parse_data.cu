#include "../include/data.h"
#include "../include/GPUErrors.h"

#define BLOCKS 16

__host__ void return_list(string path, int** arr){
    fstream data;
    data.open(path);
    string line,word;
    int count=0;
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
                    *(arr[count-1])=stoi(word);
                    arr[count-1]++;
                }
                //Extract data until ',' is found
            }
            count++;
        }
    }
    data.close();
}

__host__ void split_list(int** arr, int* subarr_1, int* subarr_2, int size){
    for(int i=0; i<size;i++){
        subarr_1[i]=arr[i][0];
        subarr_2[i]=arr[i][1];
    }
}


__global__ void Sort_Cluster(int* cluster, int* vertex, int* table, int size, int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    int idx= threadIdx.x + blockIdx.x*blockDim.x;
    int tid= threadIdx.x;
    // const int cluster_size= size/gridDim.x+1;
    extern __shared__ int shared_cluster[];
    extern __shared__ int shared_vertex[];
    extern __shared__ int bits[];
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
        bits[idx]=bit;
    }
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			bits[tid + stride] += bits[tid];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
    if(idx<size){
        int num_one_bef=bits[idx];
        int num_one_total=bits[blockDim.x-1];
        int dst = (bit==0)? (idx - num_one_bef):(size-num_one_total-num_one_bef);
        shared_vertex[dst]=vert_val;
        shared_cluster[dst]=key;
    }
    __syncthreads();
    if(idx==0){
        //Have thread 0 launch the kernel to perform the sum
        //Save the number of 0's
        table[blockIdx.x]=blockDim.x-bits[blockDim.x-1];
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=bits[blockDim.x-1];
        bit_exclusive_scan<<<1,gridDim.x,0,cudaStreamTailLaunch>>>(table);
    }
    __syncthreads();
    //We now have the pointer values in global memory to store data
    if(tid<=blockDim.x-bits[blockDim.x-1]){
        cluster[table[blockIdx.x]+tid]=shared_cluster[tid];
        vertex[table[blockIdx.x]+tid]=shared_vertex[tid];
    }
    else{
        cluster[table[blockIdx.x+gridDim.x]+tid]=shared_cluster[tid];
        vertex[table[blockIdx.x+gridDim.x]+tid]=shared_vertex[tid];
    }
}

__global__ void bit_exclusive_scan(int* bits){
    int tid = threadIdx.x;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			bits[tid + stride] += bits[tid];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
}



__host__ void Org_Vertex_Helper(int* h_cluster, int* h_vertex, int size){
    //Allocate memory for vertex and cluster info
    int* d_vertex;
    int* d_cluster;
    int* d_table;

    int threads_per_block=256;
    int blocks_per_grid= size/threads_per_block+1;

    HandleCUDAError(cudaMalloc((void**) &d_vertex, size*sizeof(int)));
    HandleCUDAError(cudaMalloc((void**) &d_cluster,size*sizeof(int)));
    HandleCUDAError(cudaMalloc((void**) &d_table,blocks_per_grid*sizeof(int)));

    HandleCUDAError(cudaMemcpy(d_vertex,h_vertex,size*sizeof(int), cudaMemcpyHostToDevice));
    HandleCUDAError(cudaMemcpy(d_cluster,h_cluster,size*sizeof(int), cudaMemcpyHostToDevice));

    for(int i=0; i<32;i++){
        Sort_Cluster<<<blocks_per_grid,threads_per_block,(2*threads_per_block+ blocks_per_grid)*sizeof(int)>>>(d_cluster,d_vertex,d_table,size,i);
        cudaDeviceSynchronize();
    }

    HandleCUDAError(cudaMemcpy(h_vertex,d_vertex,size*sizeof(int),cudaMemcpyDeviceToHost));
    HandleCUDAError(cudaMemcpy(h_cluster,d_cluster,size*sizeof(int),cudaMemcpyDeviceToHost));
    HandleCUDAError(cudaFree(d_cluster));
    HandleCUDAError(cudaFree(d_vertex));
    HandleCUDAError(cudaFree(d_table));
    HandleCUDAError(cudaDeviceReset());   
}



