#include "../include/data.h"



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
    int cluster_size= size/gridDim.x+1;
    __shared__ int shared_cluster_in[cluster_size];
    __shared__ int shared_vertex_in[cluster_size];
    __shared__ int shared_cluster_out[cluster_size];
    __shared__ int shared_vertex_out[cluster_size];
    __shared__ int bits[cluster_size];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_cluster_in[tid]=cluster[idx];
        shared_vertex_in[tid]=vertex[idx];
    }
    __syncthreads();

    //Perform sorting
    unsigned int key, bit, vert_val;
    if(idx<size){
        key=shared_cluster_in[tid];
        vert_val=shared_vertex_in[tid];
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
    //Save the number of 0's
    table[blockIdx.x]=blockDim.x-bits[blockDim.x-1];
    //Save the number of 1's
    table[blockIdx.x+gridDim.x]=bits[blockDim.x-1];
    if(idx<size){
        int num_one_bef=bits[idx];
        int num_one_total=bits[blockDim.x-1];
        int dst = (bit==0)? (idx - num_one_bef):(size-num_one_total-num_one_bef);
        shared_vertex_out[dst]=vert_val;
        shared_cluster_out[dst]=key;
    }

}


