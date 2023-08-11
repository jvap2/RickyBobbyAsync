#include "../include/data.h"
#include "../include/GPUErrors.h"

#define BLOCKS 8
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


__host__ void Check_Out_pref_sum(unsigned long int* list_1, unsigned long int* list_2, int size){
    ofstream myfile;
    myfile.open(LIST_PATH);
    myfile <<"i,List1,List2,List2Check\n";
    unsigned long int* check = new unsigned long int[size];
    check[0]=0;
    for(int i=0; i<size;i++){
        myfile<< to_string(i);
        myfile<< ",";
        if(i>0){
            check[i]=list_1[i-1]+check[i-1];
        }
        myfile<< to_string(list_1[i]);
        myfile<< ",";
        myfile<< to_string(list_2[i]);
        myfile<< ",";
        myfile<< to_string(check[i]);
        myfile<< "\n";
        if(check[i]!=list_2[i]){
            cout<<"Rugh rogh raggy, reheheheheh"<<endl;
        }
    }
    myfile.close();
    delete[] check;
}


__host__ void return_edge_list(string path, edge* arr){
    ifstream data;
    data.open(path);
    string line,word;
    unsigned long int count=0;
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

__host__ void get_graph_info(string path, unsigned long int* nodes, unsigned long int* edges){
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


__global__ void Sort_Cluster(edge* edgelist, unsigned long int* table, unsigned long int size,unsigned int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    __shared__ edge shared_edge[TPB];
    __shared__ unsigned long int bits[TPB];
    __shared__ unsigned long int ex_bits[TPB+1];
    //Load vertex and cluster info into the shared memory
    if(idx<size){
        shared_edge[tid].cluster=edgelist[idx].cluster;
        shared_edge[tid].end=edgelist[idx].end;
        shared_edge[tid].start=edgelist[idx].start;
    }
    __syncthreads();

    //Perform sorting
    unsigned long int key, bit;
    int from, to;
    if(idx<size){
        key = shared_edge[tid].cluster;
        from = shared_edge[tid].start;
        to = shared_edge[tid].end;
        bit=(key>>iter) & 0x0001;
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
        unsigned long int temp;
        if(tid>=stride){
            temp=ex_bits[tid]+ex_bits[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            ex_bits[tid]=temp;
        }
    }
    unsigned long int num_one_total;
    if(idx==size-1 || tid == blockDim.x-1){
        ex_bits[blockDim.x]=bits[tid]+ex_bits[tid];
        table[blockIdx.x]=(idx==size-1)?(size-(blockIdx.x*blockDim.x+ex_bits[blockDim.x])):(TPB-ex_bits[blockDim.x]);
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=ex_bits[blockDim.x];
    }
    __syncthreads();
    if(idx<size){
        unsigned long int num_one_bef=ex_bits[tid];
        unsigned long int num_one_total=ex_bits[blockDim.x];
        unsigned long int dst = (1-bit)*(tid - num_one_bef)+ bit*(blockDim.x-num_one_total+num_one_bef);
        shared_edge[dst].cluster=key;
        shared_edge[dst].start=from;
        shared_edge[dst].end=to;
    }
    __syncthreads();
    if(idx<size){
        edgelist[idx].cluster=shared_edge[tid].cluster;
        edgelist[idx].start=shared_edge[tid].start;
        edgelist[idx].end=shared_edge[tid].end;
    }
}

__global__ void Swap(edge* edge_list, unsigned long int* table, unsigned long int* table_2, long int size, unsigned int iter){
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    // const unsigned int cluster_size= size/gridDim.x+1;
    __shared__ edge shared_edge[TPB];
    //Load vertex and cluster info into the shared memory
    unsigned int bit, key, dst;
    if(idx<size){
        shared_edge[tid].cluster=edge_list[idx].cluster;
        shared_edge[tid].end=edge_list[idx].end;
        shared_edge[tid].start=edge_list[idx].start;
        key = shared_edge[tid].cluster;
        bit =  (key>>iter) & 1;
    }
    __syncthreads();   
    if(idx<size){
        // dst=(bit==0)?(table_2[blockIdx.x]+tid):(table_2[blockIdx.x+(gridDim.x)]+tid-(table[blockIdx.x]));
        dst = (bit==0)? (tid+table_2[blockIdx.x]):(tid-table[blockIdx.x]+table_2[blockIdx.x+gridDim.x]);
        edge_list[dst].cluster=shared_edge[tid].cluster;
        edge_list[dst].end=shared_edge[tid].end;
        edge_list[dst].start=shared_edge[tid].start;
    }
    __syncthreads();
}

__global__ void bit_exclusive_scan(unsigned long int* bits, unsigned long int* bits_2, unsigned long int* bits_3, unsigned long int size){
    unsigned int tid=threadIdx.x;
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    __shared__ unsigned int ex_bits[TPB];
    if(idx<size && idx!=0){
        ex_bits[tid]=bits[idx-1];
    }
    else{
        ex_bits[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned long int temp;
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
    if(tid==TPB-1){
        bits_3[blockIdx.x]=ex_bits[tid];
    }
    __syncthreads();
}

__global__ void fin_exclusive_scan(unsigned long int* bits_3, unsigned long int size){
    unsigned long int tid = threadIdx.x;
    unsigned long int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    __syncthreads();
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned long int temp;
        if(tid>=stride){
            temp=bits_3[tid]+bits_3[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            bits_3[tid]=temp;
        }
    }
}

__global__ void final_scan_commit(unsigned long int* bits_2, unsigned long int* bits_3, unsigned long int size){
    unsigned int bid = blockIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    if(idx<size && bid>0){
        bits_2[idx]+=bits_3[bid-1];
    }
}


//d_table_2 contains the prefix sum
//d_table contains the counts

__host__ void Org_Vertex_Helper(edge* h_edge, unsigned long int size){
    //Allocate memory for vertex and cluster info
    edge* d_edge;
    unsigned long int* d_table;
    unsigned long int* d_table_2;
    unsigned long int* d_table_3;

    unsigned long int threads_per_block=TPB;
    unsigned long int blocks_per_grid= size/threads_per_block+1;
    cout<<"Num of blocks "<<blocks_per_grid<<endl;
    unsigned long int ex_block_pg=(2*blocks_per_grid)/threads_per_block+1;
    cout<<"Second amount of blocks "<< ex_block_pg <<endl;
    
    unsigned long int* h_table=new unsigned long int[2*blocks_per_grid];
    unsigned long int* h_table_2=new unsigned long int[2*blocks_per_grid];
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge, size*sizeof(edge)))){
        cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_table,(2*blocks_per_grid)*sizeof(unsigned long int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table,0,(2*blocks_per_grid)*sizeof(unsigned long int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMalloc((void**) &d_table_2,(2*blocks_per_grid)*sizeof(unsigned long int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table_2,0,(2*blocks_per_grid)*sizeof(unsigned long int)))){
        cout<<"Unable to set table to 0"<<endl;
    }

    if(!HandleCUDAError(cudaMalloc((void**) &d_table_3,(ex_block_pg)*sizeof(unsigned long int)))){
        cout<<"Unable to allocate memory for the table data"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_table_3,0,(ex_block_pg)*sizeof(unsigned long int)))){
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
    if(ex_block_pg>0){
        for(unsigned int i=0; i<(unsigned int)log2(double(BLOCKS));i++){
            Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,size,i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host with Sort Cluster"<<endl;
            }
            bit_exclusive_scan<<<ex_block_pg,threads_per_block>>>(d_table,d_table_2,d_table_3,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host exclusive scan"<<endl;
            }
            fin_exclusive_scan<<<1,ex_block_pg,sizeof(int)*ex_block_pg>>>(d_table_3,ex_block_pg);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host for final exclusive scan"<<endl;
            }
            final_scan_commit<<<ex_block_pg,threads_per_block>>>(d_table_2,d_table_3,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host for final exclusive scan commit"<<endl;
            }
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,d_table_2,size, i);
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
            bit_exclusive_scan<<<ex_block_pg,threads_per_block>>>(d_table,d_table_2,d_table_3,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host exclusive scan"<<endl;
            }
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,d_table_2,size, i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
        }
    }


    if(!HandleCUDAError(cudaMemcpy(h_edge,d_edge,size*sizeof(edge),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back edge data"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_table,d_table,2*blocks_per_grid*sizeof(unsigned long int),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back edge data"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(h_table_2,d_table_2,2*blocks_per_grid*sizeof(unsigned long int),cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy back edge data"<<endl;
    }

    Check_Out_pref_sum(h_table,h_table_2,2*blocks_per_grid);

    delete[] h_table;
    delete[] h_table_2;

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