#include "../include/data.h"
#include "../include/GPUErrors.h"

#define BLOCKS 16
#define TPB 256




__host__ void Check_Out_csv_edge(edge* edge_list, int size){
    ofstream myfile;
    myfile.open(CLUSTER_PATH);
    myfile <<"from,to,cluster\n";
    for(int i=0; i<size;i++){
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
    cout<<count<<endl;
    data.close();
}

__host__ void get_graph_info(string path, int* nodes, int* edges){
    ifstream data;
    data.open(path);
    string line,word;
    int count =0;
    int column = 0;
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); 
            while(getline(stream_data,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        cout<<word<<endl;
                        *nodes=stoi(word);
                        column++;
                    }
                    else{
                        *edges=stoi(word);
                    }
                }
                //Extract data until ',' is found
            }
            count++;
        }
    }

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
        shared_edge[tid].cluster=edgelist[idx].cluster;
        shared_edge[tid].end=edgelist[idx].end;
        shared_edge[tid].start=edgelist[idx].start;
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
    __syncthreads();
    unsigned num_one_total;
    if(idx<size){
        unsigned int num_one_bef=ex_bits[tid];
        unsigned int num_one_total=ex_bits[blockDim.x-1]+bits[blockDim.x-1];
        unsigned int dst = (bit==0)? (tid - num_one_bef):(blockDim.x-num_one_total+num_one_bef-1);
        if(dst<0 || dst>blockDim.x){
            printf("%d \n",dst);
        }
        shared_edge[dst].cluster=key;
        shared_edge[dst].start=from;
        shared_edge[dst].end=to;
    }
    __syncthreads();
    if(tid==0){
        table[blockIdx.x]=blockDim.x-num_one_total;
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=num_one_total;
    }
    __syncthreads();
    if(idx<size){
        edgelist[idx].cluster=shared_edge[tid].cluster;
        edgelist[idx].start=shared_edge[tid].start;
        edgelist[idx].end=shared_edge[tid].end;
    }
}

__global__ void Swap(edge* edge_list, unsigned int* table, unsigned  int size, unsigned int iter){
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    // const unsigned int cluster_size= size/gridDim.x+1;
    __shared__ edge shared_edge[TPB];
    //Load vertex and cluster info into the shared memory
    unsigned int bit, key;
    if(idx<size){
        shared_edge[tid].cluster=edge_list[idx].cluster;
        shared_edge[tid].end=edge_list[idx].end;
        shared_edge[tid].start=edge_list[idx].start;
        key = shared_edge[tid].cluster;
        bit =  (key>>iter) & 1;
    }
    __syncthreads();   
    if(idx<size){
        edge_list[table[blockIdx.x+(gridDim.x*bit)]+tid].cluster=shared_edge[tid].cluster;
        edge_list[table[blockIdx.x+(gridDim.x*bit)]+tid].end=shared_edge[tid].end;
        edge_list[table[blockIdx.x+(gridDim.x*bit)]+tid].start=shared_edge[tid].start;
    }
    __syncthreads();
}

__global__ void bit_exclusive_scan(unsigned int* bits, unsigned int* bits_2, unsigned int size){
    unsigned int tid=threadIdx.x;
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    __shared__ unsigned int ex_bits[TPB];
    if(idx<size && tid!=0){
        ex_bits[tid]=bits[idx-1];
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
        bits_2[idx]=ex_bits[tid];
    }
    __syncthreads();
}

__global__ void fin_exclusive_scan(unsigned int* bits_2, unsigned int* bits_3, unsigned int size){
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    extern __shared__ unsigned int s_bit[];
    if(idx<size && tid!=0){
        s_bit[tid]=bits_2[(idx)*TPB-1];
    }
    else{
        s_bit[tid]=0;
    }
    __syncthreads();
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=s_bit[tid]+s_bit[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            s_bit[tid]=temp;
        }
    }
    if(idx<size){
        bits_3[idx]=s_bit[tid];
    }
}

__global__ void final_scan_commit(unsigned int* bits_2, unsigned int* bits_3, unsigned int size){
    unsigned int bid = blockIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    if(idx<size){
        bits_2[idx]+=bits_3[bid];
    }
}


//d_table_2 contains the prefix sum
//d_table contains the counts

__host__ void Org_Vertex_Helper(edge* h_edge, int size){
    //Allocate memory for vertex and cluster info
    edge* d_edge;
    unsigned int* d_table;
    unsigned int* d_table_2;
    unsigned int* d_table_3;

    unsigned int threads_per_block=TPB;
    unsigned int blocks_per_grid= size/threads_per_block+1;
    cout<<"Num of blocks "<<blocks_per_grid<<endl;
    unsigned int ex_block_pg=(2*blocks_per_grid)/threads_per_block+1;
    cout<<"Second amount of blocks "<< ex_block_pg <<endl;
    
    // unsigned int* h_table=new unsigned int[2*blocks_per_grid];
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge, size*sizeof(edge)))){
        cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_table,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table,0,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMalloc((void**) &d_table_2,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table_2,0,(2*blocks_per_grid)*sizeof(unsigned int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMalloc((void**) &d_table_3,(ex_block_pg)*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table_3,0,(ex_block_pg)*sizeof(unsigned int)))){
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
    if(ex_block_pg>1){
        for(unsigned int i=0; i<=(unsigned int)log2((double)BLOCKS);i++){
            Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,size,i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host with Sort Cluster"<<endl;
            }
            bit_exclusive_scan<<<ex_block_pg,threads_per_block>>>(d_table,d_table_2,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host exclusive scan"<<endl;
            }
            fin_exclusive_scan<<<1,ex_block_pg,sizeof(int)*ex_block_pg>>>(d_table_2,d_table_3,ex_block_pg);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host for final exclusive scan"<<endl;
            }
            final_scan_commit<<<ex_block_pg,threads_per_block>>>(d_table_2,d_table_3,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host for final exclusive scan commit"<<endl;
            }
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table_2,size, i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
        }
    }
    else{
        for(unsigned int i=0; i<32;i++){
            Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,size,i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host with Sort Cluster"<<endl;
            }
            bit_exclusive_scan<<<ex_block_pg,threads_per_block>>>(d_table,d_table_2,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host exclusive scan"<<endl;
            }
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table_2,size, i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
        }
    }


    if(!HandleCUDAError(cudaMemcpy(h_edge,d_edge,size*sizeof(edge),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back edge data"<<endl;
    }
    // if(!HandleCUDAError(cudaMemcpy(h_table,d_table_2,2*blocks_per_grid*sizeof(int),cudaMemcpyDeviceToHost))){
    //     cout<<"Unable to copy back edge data"<<endl;
    // }
    // cout<<"exclusive sum table"<<endl;
    // for(int i = 0 ; i< 100; i++ ){
    //     cout<<h_table[i]<<endl;
    // }
    // cout<<"Done with exclusive sum table"<<endl;

    // delete[] h_table;

    HandleCUDAError(cudaFree(d_edge));
    HandleCUDAError(cudaFree(d_table));
    HandleCUDAError(cudaFree(d_table_2));
    HandleCUDAError(cudaFree(d_table_3));
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