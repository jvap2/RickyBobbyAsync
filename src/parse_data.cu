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


__global__ void Sort_Cluster(int* cluster, int* vertex, int size, int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    int idx= threadIdx.x + blockIdx.x*blockDim.x;
    int tid= threadIdx.x;
    int cluster_size= size/gridDim.x+1;
    __shared__ int shared_cluster[cluster_size];
    __shared__ int shared_vertex[cluster_size];
    //Load vertex and cluster info into the shared memory

    shared_cluster[tid]=cluster[idx];
    shared_vertex[tid]=vertex[idx];
    __syncthreads();

    //Perform sorting



}


