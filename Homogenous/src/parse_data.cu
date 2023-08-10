#include "../include/data.h"
#include "../include/GPUErrors.h"

#define BLOCKS 16
#define TPB 256




__host__ void Check_Out_csv_edge(edge* edge_list){
    ofstream myfile;
    myfile.open(CLUSTER_PATH);
    myfile <<"from,to,cluster\n";
    for(int i=0; i<EDGES;i++){
        myfile<< to_string(edge_list[i].start);
        myfile<< ",";
        myfile<< to_string(edge_list[i].end);
        myfile<< ",";
        myfile<< to_string(edge_list[i].cluster);
        myfile<< "\n";
    }
    myfile.close();
}

__host__ void return_edge_list(string path, edge* arr){
    ifstream data;
    data.open(path);
    string line,word;
    unsigned int count=0;
    unsigned int column=0;
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
                    if(column==0){
                        arr[count-1].start=stoi(word);
                        column++;
                    }
                    else{
                        arr[count-1].end=stoi(word);
                        arr[count-1].cluster=0;
                    }
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


__host__ void split_list(unsigned int** arr, unsigned int* subarr_1, unsigned int* subarr_2, unsigned int size){
    for(unsigned int i=0; i<size;i++){
        subarr_1[i]=arr[i][0];
        subarr_2[i]=arr[i][1];
    }
}


__global__ void Sort_Cluster(edge* edgelist, unsigned int* table, unsigned int size,unsigned int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    __shared__ edge shared_edge[TPB];
    __shared__ unsigned int bits[TPB];
    __shared__ unsigned int ex_bits[TPB];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_edge[tid]=edgelist[idx];
    }
    __syncthreads();

    //Perform sorting
    unsigned int key, bit;
    int from, to;
    if(idx<size){
        key = shared_edge[tid].cluster;
        from = shared_edge[tid].start;
        to = shared_edge[tid].end;
        bit=(key>>iter) & 1;
        bits[tid]=bit;
    }
    __syncthreads();
    //Perform exclusive scan
    if(idx<size && tid!=0){
        ex_bits[tid]=bits[tid-1];
    }
    else{
        ex_bits[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=ex_bits[tid]+ex_bits[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            ex_bits[tid]=temp;
        }
    }
    if(idx<size){
        bits[tid]=ex_bits[tid];
    }
    __syncthreads();
    unsigned num_one_total;
    if(idx<size){
        unsigned int num_one_bef=bits[tid];
        unsigned int num_one_total=bits[TPB-1];
        unsigned int dst = (bit==0)? (tid - num_one_bef):(TPB-num_one_total+num_one_bef-1);
        shared_edge[dst].cluster=key;
        shared_edge[dst].start=from;
        shared_edge[dst].end=to;
    }
    __syncthreads();
    if(tid==0){
        table[blockIdx.x]=blockDim.x-bits[blockDim.x-1];
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=bits[blockDim.x-1];
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
        if(tid<TPB-num_one_total){
            edgelist[table[blockIdx.x]+tid].cluster=shared_edge[tid].cluster;
            edgelist[table[blockIdx.x]+tid].start=shared_edge[tid].start;
            edgelist[table[blockIdx.x]+tid].end=shared_edge[tid].end;
        }
        else{
            edgelist[table[blockIdx.x+gridDim.x]+tid].cluster=shared_edge[tid].cluster;
            edgelist[table[blockIdx.x+gridDim.x]+tid].start=shared_edge[tid].start;
            edgelist[table[blockIdx.x+gridDim.x]+tid].end=shared_edge[tid].end;
        }
    }
    __syncthreads();
}

__global__ void Swap(unsigned int* cluster, unsigned int* vertex, unsigned int* table, unsigned int* table_2, unsigned  int size){
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    // const unsigned int cluster_size= size/gridDim.x+1;
    __shared__ unsigned int shared_edge[TPB];
    __shared__ unsigned int shared_vertex[TPB];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_edge[tid]=cluster[idx];
        shared_vertex[tid]=vertex[idx];
    }
    __syncthreads();   
    if(idx<size){
        if(tid<=table_2[blockIdx.x]){
            cluster[table[blockIdx.x]+tid]=shared_edge[tid];
            vertex[table[blockIdx.x]+tid]=shared_vertex[tid];
        }
        else{
            cluster[table[blockIdx.x+gridDim.x]+tid]=shared_edge[tid];
            vertex[table[blockIdx.x+gridDim.x]+tid]=shared_vertex[tid];
        }
    }
    __syncthreads();
}

__global__ void bit_exclusive_scan(unsigned int* bits,unsigned int size){
    unsigned int tid=threadIdx.x;
    __shared__ unsigned int ex_bits[TPB];
    if(tid<size && tid!=0){
        ex_bits[tid]=bits[tid-1];
    }
    else{
        ex_bits[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=ex_bits[tid]+ex_bits[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            ex_bits[tid]=temp;
        }
    }
    if(tid<TPB){
        // bit_2[tid]=ex_bits[tid];
        bits[tid]=ex_bits[tid];
    }
    __syncthreads();
}



__host__ void Org_Vertex_Helper(edge* h_edge, int size){
    //Allocate memory for vertex and cluster info
    edge* d_edge;
    unsigned int* d_table;
    // unsigned int* d_table_2;

    unsigned int threads_per_block=TPB;
    unsigned int blocks_per_grid= size/threads_per_block+1;

    if(!HandleCUDAError(cudaMalloc((void**) &d_edge, size*sizeof(edge)))){
        cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_table,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table,0,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMemcpy(d_edge,h_edge,size*sizeof(edge), cudaMemcpyHostToDevice))){
        cout<<"Unable to copy cluster data"<<endl;
    }
    double r = ((double) rand() / (RAND_MAX));
    Random_Edge_Placement<<<blocks_per_grid,threads_per_block>>>(d_edge, r);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Rand_Edge Place"<<endl;
    } 
    for(unsigned int i=0; i<=(unsigned int)log2((double)BLOCKS);i++){
        Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,size,i);
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

    if(!HandleCUDAError(cudaMemcpy(h_edge,d_edge,size*sizeof(unsigned int),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back edge data"<<endl;
    }
    HandleCUDAError(cudaFree(d_edge));
    HandleCUDAError(cudaFree(d_table));
    HandleCUDAError(cudaDeviceReset());   
}



__host__ graph *create_graph (edge *edges){
   int i;
   struct graph *graph = (struct graph *) malloc (sizeof (struct graph));
   for (i = 0; i < NODES; i++) {
      graph->point[i] = NULL;
   }
   for (i = 0; i < EDGES; i++) {
      int start = edges[i].start;
      int end = edges[i].end;
      struct vertex *v = (struct vertex *) malloc (sizeof (struct vertex));
      v->end = end;
      v->next = graph->point[start];
      graph->point[start] = v;
   }
   return graph;
}

__global__ void Random_Edge_Placement(edge *edges, double rand_num){
    unsigned int idx= threadIdx.x+blockDim.x*blockIdx.x;
    __syncthreads();
    //Use multiplication hashing
    double intpart;
    double mod_part = modf(idx*rand_num, &intpart);
    unsigned int hash = (unsigned int)(BLOCKS*mod_part);
    //We now have the key, we need to sort
    if(idx<EDGES){
        edges[idx].cluster=hash;
    }
    __syncthreads();

}