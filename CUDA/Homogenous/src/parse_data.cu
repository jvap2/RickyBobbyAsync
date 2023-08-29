#include "../include/data.h"
#include "../include/GPUErrors.h"





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

__host__ void Check_Out_ptr(unsigned int* edge_list, int size){
    ofstream myfile;
    myfile.open(PTR_PATH);
    myfile <<"ptr\n";
    for(int i=0; i<size;i++){
        myfile<< to_string(edge_list[i]);
        myfile<< "\n";
    }
    myfile.close();
}


__host__ void Check_Out_pref_sum(unsigned int* list_1, unsigned int* list_2, int size){
    ofstream myfile;
    myfile.open(LIST_PATH);
    myfile <<"i,List1,List2,List2Check\n";
    unsigned int* check = new unsigned int[size];
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


__host__ void check_out_replicas(string path,unsigned int* replicas, unsigned int node_size){
    unsigned int total_rep;
    float rep_avg;
    total_rep=0;
    rep_avg=0;
    for(int i=0; i<node_size;i++){
        total_rep+=replicas[i];
    }
    rep_avg=1.0f*total_rep/(1.0f*node_size);
    ofstream myfile;
    myfile.open(path, ios::app);
    myfile<< to_string(node_size);
    myfile<< ",";
    myfile<< to_string(total_rep);
    myfile<< ",";
    myfile<< to_string(rep_avg);
    myfile<< ",";
    myfile<< to_string(BLOCKS);
    myfile<< "\n";
    myfile.close();
}


__host__ void return_edge_list(string path, edge* arr){
    cout<<"Getting edge list"<<endl;
    ifstream data;
    data.open(path);
    string line,word;
    unsigned int count=0;
    unsigned int column=0;
    cout<<"Data is open "<<data.is_open()<<endl;
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
                        arr[count-1].start=stoul(word);
                        arr[count-1].local_start=0u;
                        column++;
                    }
                    else{
                        arr[count-1].end=stoul(word);
                        arr[count-1].local_end=0u;
                        arr[count-1].cluster=0u;
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


__host__ void CSR_Graph(string path, unsigned int node_size, unsigned int edge_size, unsigned int* src_ptr, unsigned int* succ, unsigned int* deg_arr){
    ifstream data;
    data.open(path);
    string line,word;
    unsigned int count = 0;
    unsigned int column=0;
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
                        src_ptr[stoi(word)]++; //Create a histogram of values
                        deg_arr[stoi(word)]++;
                        column++;
                    }
                    else{
                        succ[count-1]=stoul(word);

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
    //Perform prefix sum of src_prt
    unsigned int* copy_ptr = new unsigned int[node_size+1];
    copy_ptr[0]=0;
    for(unsigned int i=1; i<node_size+1;i++){
        copy_ptr[i]=src_ptr[i-1];
    }
    src_ptr[0]=0;
    for(unsigned int i=1; i<node_size+1;i++){
        copy_ptr[i]+=copy_ptr[i-1];
        src_ptr[i]=copy_ptr[i];
    }
    cout<<count<<endl;
    data.close();
}

__host__ void Capture_Node_Degree(edge* edge_list, unsigned int* deg_arr, unsigned int size){
    for(unsigned int i=0; i<size;i++){
        deg_arr[edge_list[i].start]++;
    }
}

__host__ void get_graph_info(string path, unsigned int* nodes, unsigned int* edges){
    cout<<"Getting graph info"<<endl;
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


// C++ implementation of Radix Sort


// A utility function to get maximum
// value in arr[]
__host__ int getMax_cluster(edge* edge_list, int n)
{
    int mx = edge_list[0].cluster;
    for (int i = 1; i < n; i++)
        if (edge_list[i].cluster > mx)
            mx = edge_list[i].cluster;
    return mx;
}

// A function to do counting sort of arr[]
// according to the digit
// represented by exp.
__host__ void cpu_countSort(edge* arr, int n, int exp)
{

    // Output array
    edge* out;
    out=(edge*)malloc(sizeof(edge)*n);
    int i, count[10] = { 0 };

    // Store count of occurrences
    // in count[]
    for (i = 0; i < n; i++)
        count[(arr[i].cluster / exp) % 10]++;

    // Change count[i] so that count[i]
    // now contains actual position
    // of this digit in output[]
    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    // Build the output array
    for (i = n - 1; i >= 0; i--) {
        out[count[(arr[i].cluster / exp) % 10] - 1] = arr[i];
        count[(arr[i].cluster / exp) % 10]--;
    }

    // Copy the output array to arr[],
    // so that arr[] now contains sorted
    // numbers according to current digit
    for (i = 0; i < n; i++)
        arr[i] = out[i];

    free(out);
}

// The main function to that sorts arr[]
// of size n using Radix Sort
__host__ void cpu_radixsort(edge* arr, int n)
{

    // Find the maximum number to
    // know number of digits
    int m = getMax_cluster(arr, n);

    // Do counting sort for every digit.
    // Note that instead of passing digit
    // number, exp is passed. exp is 10^i
    // where i is current digit number
    for (int exp = 1; m / exp > 0; exp *= 10)
        cpu_countSort(arr, n, exp);
}


__global__ void Sort_Cluster(edge* edgelist, unsigned int* table, unsigned int size,unsigned int iter){
    //Need to sort through the cluster data and organize it
    //organize into the data for each block of FrogWild
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    __shared__ edge shared_edge[TPB];
    __shared__ unsigned int bits[TPB];
    __shared__ unsigned int ex_bits[TPB+1];
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
    unsigned int num_one_total;
    if(idx==size-1 || tid == blockDim.x-1){
        ex_bits[blockDim.x]=bits[tid]+ex_bits[tid];
        //Save the number of 0's
        table[blockIdx.x]=(idx==size-1)?(size-(blockIdx.x*blockDim.x+ex_bits[blockDim.x])):(TPB-ex_bits[blockDim.x]);
        //Save the number of 1's
        table[blockIdx.x+gridDim.x]=ex_bits[blockDim.x];
    }
    __syncthreads();
    if(idx<size){
        unsigned int num_one_bef=ex_bits[tid];
        unsigned int num_one_total=ex_bits[blockDim.x];
        unsigned int dst =(bit==0)?(tid-num_one_bef):(blockDim.x-num_one_total+num_one_bef);
        if(dst>TPB-1 || dst<0){
            printf("Error: %d, %d, %d, %d, %d, %d, %d\n", idx, tid, blockIdx.x, bit, num_one_bef, num_one_total, dst);
        }
        // unsigned int dst = (1-bit)*(tid - num_one_bef)+ bit*(blockDim.x-num_one_total+num_one_bef);
        shared_edge[dst].cluster=key;
        shared_edge[dst].start=from;
        shared_edge[dst].end=to;
    }
    __syncthreads();
    if(idx<size){
        edgelist[idx]=shared_edge[tid];
        //The edge list is now sorted block-wise
    }
}

__global__ void Swap(edge* edge_list, edge* edge_list_2, unsigned int* table, unsigned int* table_2, long int size, unsigned int iter){
    unsigned int idx= threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid= threadIdx.x;
    // const unsigned int cluster_size= size/gridDim.x+1;
    __shared__ edge shared_edge[TPB];
    //Load vertex and cluster info into the shared memory
    unsigned int bit, key, dst;
    if(idx<size){
        shared_edge[tid]=edge_list[idx];
        key = shared_edge[tid].cluster;
        bit =  (key>>iter) & 1;
    }
    __syncthreads();   
    if(idx<size){
        dst = (bit==0)? (tid+table_2[blockIdx.x]):(tid-table[blockIdx.x]+table_2[blockIdx.x+gridDim.x]);
        edge_list_2[dst]=shared_edge[tid];
    }
}

__global__ void bit_exclusive_scan(unsigned int* bits, unsigned int* bits_2, unsigned int* bits_3, unsigned int size){
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
    if(tid==TPB-1){
        bits_3[blockIdx.x]=ex_bits[tid];
    }
    __syncthreads();
}

__global__ void fin_exclusive_scan(unsigned int* bits_3, unsigned int size){
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    __syncthreads();
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=bits_3[tid]+bits_3[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            bits_3[tid]=temp;
        }
    }
}

__global__ void final_scan_commit(unsigned int* bits_2, unsigned int* bits_3, unsigned int size){
    unsigned int bid = blockIdx.x;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    if(idx<size && bid>0){
        bits_2[idx]+=bits_3[bid-1];
    }
}


//d_table_2 contains the prefix sum
//d_table contains the counts
__global__ void copy_edge_list(edge* edge_1, edge* edge_2, unsigned int size){
    unsigned int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        edge_1[idx]=edge_2[idx];
    }
}


__global__ void Random_Edge_Placement(edge *edges, double rand_num, unsigned int size){
    unsigned int idx= threadIdx.x+blockDim.x*blockIdx.x;
    __syncthreads();
    //Use multiplication hashing
    double intpart;
    double mod_part = modf(idx*rand_num, &intpart);
    unsigned int hash = (unsigned int)(BLOCKS*mod_part);
    //We now have the key, we need to sort
    if(idx<size){
        edges[idx].cluster=hash;
    }
    __syncthreads();

}


/*CHECK THIS ONE- MAKE SURE THE CSR FORMAT IS PROPER*/
__global__ void Degree_Based_Placement(edge* edges, unsigned int* deg_arr, double rand_num, replica_tracker* d_rep, unsigned int size){
    unsigned int idx= threadIdx.x+blockDim.x*blockIdx.x;
    if(idx<size){
        unsigned int start = edges[idx].start;
        unsigned int end = edges[idx].end;
        unsigned int deg_start = deg_arr[start];
        unsigned int deg_end = deg_arr[end];
        unsigned int v_hash = (deg_start>deg_end)?start:end;
        double intpart;
        double mod_part = modf(v_hash*rand_num, &intpart);
        unsigned int hash = (unsigned int)floor(BLOCKS*mod_part);
        edges[idx].cluster=hash;
        //Now, we need to update the replica tracker
        /*We are going to need to use some atomic form to be able to write correctly*/
        atomicOr(&d_rep[start].clusters[hash],1);
        atomicOr(&d_rep[end].clusters[hash],1);

    }

}


/*We will now need to reduce the d_rep stuff*/

__global__ void Finalize_Replica_Tracker(replica_tracker* d_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ replica_tracker shared_rep[TPB];
    if(idx<node_size){
        shared_rep[tid]=d_rep[idx];
    }
    __syncthreads();
    if(idx<node_size){
        for(int i=0; i<BLOCKS; i++){
            if(shared_rep[tid].clusters[i]==1){
                shared_rep[tid].clusters[i]=1;
                shared_rep[tid].num_replicas++;
            }
        }
    }
    __syncthreads();
    if(idx<node_size){
        d_rep[idx]=shared_rep[tid];
    }
}


__global__ void Histogram_1(edge* edgelist, unsigned int* hist_bin, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int s_edge_list[TPB];
    __shared__ unsigned int s_hist[BLOCKS];
    if(idx<size){
        s_edge_list[tid]=edgelist[idx].cluster;
        //Copy TPB cluster values over
    }
    __syncthreads();
    if(idx<size){
        atomicAdd(&s_hist[s_edge_list[tid]],1);
    }
    __syncthreads();
    if(tid<BLOCKS){
        hist_bin[BLOCKS*blockIdx.x+tid]=s_hist[tid];
    }
    //Now, all the data is stored locally on a blocks/grid by BLOCKS array which we need to reduce
}

__global__ void Kogge_Stone_Hist_Reduct(unsigned int* hist_bin, unsigned int* fin_bin, int size){
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int clust_val[];
    if(idx<size){
        clust_val[tid]=hist_bin[tid*BLOCKS+blockIdx.x];
    }
    else{
        clust_val[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=clust_val[tid]+clust_val[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            clust_val[tid]=temp;
        }
    }
    if(tid==blockDim.x){
        fin_bin[blockIdx.x]=clust_val[tid];
    }
    __syncthreads();
}

__global__ void Hist_Prefix_Sum(unsigned int* fin_bin, unsigned int* fin_bin_2){
    unsigned int tid = threadIdx.x+blockDim.x*blockIdx.x;
    __shared__ unsigned int local[BLOCKS];
    if(tid<BLOCKS && tid!=0){
        local[tid]=fin_bin[tid-1];
    }
    else{
        local[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=local[tid]+local[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            local[tid]=temp;
        }
    }
    if(tid<BLOCKS){
        fin_bin_2[tid]=local[tid];
    }
}

__global__ void First_Init(float* rand_frog, unsigned int* d_frog, unsigned int node_size, unsigned int edge_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<node_size/20){
        rand_frog[idx]=floorf(rand_frog[idx]*node_size);
        atomicAdd(&d_frog[(int)rand_frog[idx]],1);
    }
}


/*
What we need for the iterations of pagerank:
(1)Gather
(2)Apply
(3)Scatter
---------------------------------------------
(1) Gather: 
-First time, initialize random frogs (done)
-Remaining iterations, we need to collect the frogs from the previous iteration sent to nodes from scatter

(2) Apply:
-This function takes care for keeping track of the number of frogs that have stopped on each vertex

(3)Scatter 
-This function takes care of sending the frogs to the next vertex

Instead of dictating which block is the master of which vertex, we will have the global memory act as the sole master
of the vertex. This will allow us to combine the functions into one and avoid passing of data, and ease the synchronization
*/


__global__ void FrogWild(edge* edgelist, unsigned int* d_src, unsigned int* d_succ,
unsigned int* d_c, unsigned int* d_frogs, unsigned int* edge_ptr,unsigned int* ctr_ptr,
unsigned int size, unsigned int iter){
    /*since init is done first with first init, we begin with apply, scatter,
    and then we will iterate through the rest*/
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int local_frogs[];
    extern __shared__ unsigned int local_c[];
    extern __shared__ unsigned int local_src[];
    extern __shared__ unsigned int local_succ[];
    extern __shared__ edge local_edge[];
    //Fetch memory for the block
    if(idx<size){
        //Use thread coarsening
        for(int i = tid; i<ctr_ptr[blockIdx.x];i+=blockDim.x){
            local_edge[i]=edgelist[edge_ptr[blockIdx.x]+i];
        }
    }
    //We have fetched the local edges now, we need to gather the frogs, vertices etc for the block
    __syncthreads();
    if(idx<size){
        for(int i = tid; i<ctr_ptr[blockIdx.x];i+=blockDim.x){
            local_c[i]=d_c[local_edge[i].start];
            local_src[i]=d_src[local_edge[i].start];
            local_succ[i]=d_succ[local_edge[i].start];
        }
    }


}


__global__ void gen_backward_mask(unsigned int* global_list, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* start_mask, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int start[];
    extern __shared__ unsigned int start_back_mask[];
    if(idx<size){
        //Check that the ctr table is doing what we want
        for(int i=tid; i<2*ctr_table[blockIdx.x];i+=blockDim.x){
            start[i]=global_list[2*ptr_table[blockIdx.x]+i];
        }
    }
    __syncthreads();
    if(idx<size){
        /*Now, we need to generate the hash values*/
        /*We will utilize run length encoding to find the unique values*/
        for(int i = tid; i<2*ctr_table[blockIdx.x];i+=blockDim.x){
            if(i==0){
                start_back_mask[i]=1;
            }
            else{
                if(start[i]!=start[i-1]){
                    start_back_mask[i]=1;
                }
                else{
                    start_back_mask[i]=0;
                }
            }
        }
    }
    __syncthreads();
    /*We have the mask, now, we need to commit to global memory for next kernel*/
    for(int i=tid; i<2*ctr_table[blockIdx.x];i+=blockDim.x){
        start_mask[2*ptr_table[blockIdx.x]+i]=start_back_mask[i];
    }
}


__global__ void scan_mask(unsigned int* start_mask, unsigned* compct_start, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    //We need to use global memory if we intend to use dynamic parallelism, so we need to copy the data over
    /*Now, we can execute the exclusive scan- issue will be that this will be larger than the size of a thread block
    Can we use dynamic parallelism in order to compute partial sums to then acquire a final sum?*/
    int num_of_blocks = (ctr_table[blockIdx.x]/blockDim.x)+1;
    if(tid<num_of_blocks){
        int dym_size=(tid==num_of_blocks-1)?(ctr_table[blockIdx.x]-tid*blockDim.x):(blockDim.x);
        Prefix_Scan_Cmpt<<<1,TPB>>>(start_mask+ptr_table[blockIdx.x]+tid*blockDim.x, compct_start+ptr_table[blockIdx.x]+tid*blockDim.x,dym_size,tid);
    }
    __syncthreads();
    extern __shared__ unsigned int end_vals[];
    /*Now, we have partial sums, we need to find the final value of each accumulated sum*/
    /*How do we do this..... SUBLIME!*/
    if(tid<num_of_blocks){
        int loc = (tid==num_of_blocks-1)? (ptr_table[blockIdx.x+1]-1):(ptr_table[blockIdx.x]+(tid+1)*blockDim.x-1);
        end_vals[tid]=compct_start[loc];
    }
    __syncthreads();
    if(tid<num_of_blocks){
        int dym_size=(tid==num_of_blocks-1)?(num_of_blocks-tid):(num_of_blocks);
        //Check this
        Prefix_Scan_Cmpt<<<1,num_of_blocks>>>(end_vals, end_vals,dym_size,0);
    }
    __syncthreads();
    if(tid<num_of_blocks){
        int dym_size=(tid==num_of_blocks-1)?(num_of_blocks-tid):(num_of_blocks);
        final_scan_commit<<<1,TPB>>>(compct_start+ptr_table[blockIdx.x]+tid*blockDim.x,end_vals, dym_size);
    }
    
}


__global__ void Prefix_Scan_Cmpt(unsigned int* mask, unsigned int* cmpt, unsigned int size, unsigned int block){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_cmpt[TPB];
    if(block==0){
        if(tid<size && tid!=0){
            local_cmpt[tid]=mask[tid-1];
        }
        else{
            local_cmpt[tid]=0;
        }
    }
    else{
        if(tid<size){
            local_cmpt[tid]=mask[tid];
        }

    }
    __syncthreads();
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=local_cmpt[tid]+local_cmpt[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            local_cmpt[tid]=temp;
        }
    }
    if(idx<size){
        cmpt[idx]=local_cmpt[tid];
    }
    __syncthreads();
}


/*CHECK THIS*/

__global__ void Scanned_To_Compact(unsigned int* cmpt, unsigned int* scanned, unsigned int* new_size, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<size){
        for(int i = tid; i<ctr_table[blockIdx.x];i+=blockDim.x){
            if(i==0){
                cmpt[ptr_table[blockIdx.x]+i]=0;
            }
            if(i==size-1){
                cmpt[ptr_table[blockIdx.x]+scanned[ptr_table[blockIdx.x]+i]]=i+1;
                *(new_size+blockIdx.x)=scanned[ptr_table[blockIdx.x]+i];
            }
            else if(scanned[ptr_table[blockIdx.x]+i]!=scanned[ptr_table[blockIdx.x]+i-1]){
                cmpt[scanned[ptr_table[blockIdx.x]+i]]=i;
            }
        }
    }
}

__global__ void Final_Compression(unsigned int* cmpt, unsigned int* new_size, unsigned int* in, unsigned int* new_idx, unsigned int* out){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    for(int i = tid; i<(*new_size+blockIdx.x);i+=blockDim.x){
        out[i]=in[cmpt[i]];
        new_idx[i]=cmpt[i+1]-cmpt[i];
    } 
}


__global__ void acc_accum(unsigned int* approx, unsigned int* pagerank, unsigned int* table, unsigned int k){
    unsigned int idx=threadIdx.x + (blockIdx.x*blockDim.x);
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_table[TPB];
    if(idx<k){
        if(approx[idx]==pagerank[idx]){
            local_table[tid]=1;
        }
        else{
            local_table[tid]=0;
        }
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=local_table[tid]+local_table[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            local_table[tid]=temp;
        }
    }
    if(tid==(blockDim.x-1)){
        local_table[tid]=table[idx];
    }
}

__global__ void fin_acc(unsigned int* table, unsigned int k, float* acc){
    unsigned int tid = threadIdx.x;
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=table[tid]+table[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            table[tid]=temp;
        }
    }
    if(tid==(blockDim.x-1)){
        *acc=float(table[tid])/float(k);
    }
}

__global__ void Find_Max_Cluster(unsigned int* ctr_table, unsigned int* max_val){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ unsigned int local_max[BLOCKS];
    if(idx<BLOCKS){
        local_max[tid]=ctr_table[idx];
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride && (tid + 1)%(stride*2)==0){
            temp=(local_max[tid]>local_max[tid-stride])?local_max[tid]:local_max[tid-stride];
        }
        __syncthreads();
        if(tid>=stride && (tid + 1)%(stride*2)==0){
            local_max[tid]=temp;
        }
    }
    __syncthreads();
    if(tid==blockDim.x-1){
        *max_val=local_max[tid-1];
    }
}

__global__ void unq_exclusive_scan(unsigned int* len, unsigned int* unq_ptr){
    unsigned int tid=threadIdx.x;
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    __shared__ unsigned int local_ptr_val[BLOCKS];
    if(idx<BLOCKS && idx!=0){
        local_ptr_val[tid]=len[idx-1];
    }
    else{
        local_ptr_val[tid]=0;
    }
    for(unsigned int stride = 1; stride<blockDim.x;stride*=2){
        __syncthreads();
        unsigned int temp;
        if(tid>=stride){
            temp=local_ptr_val[tid]+local_ptr_val[tid-stride];
        }
        __syncthreads();
        if(tid>=stride){
            local_ptr_val[tid]=temp;
        }
    }
    if(idx<BLOCKS){
        unq_ptr[idx]=local_ptr_val[tid];
    }
    __syncthreads();
}


__global__ void Find_Length_of_Unique(unsigned int* start_len, unsigned int* end_len, unsigned int* vector_length){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    __shared__ unsigned int local_size[BLOCKS];
    if(idx<BLOCKS){
        local_size[idx]=start_len[idx]+end_len[idx];
    }

}


__global__ void Naive_Merge_Sort(unsigned int* start, unsigned int* end, unsigned int* ptr_table, unsigned int* ctr_table, unsigned int* unq){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    unsigned int tid = threadIdx.x;
    unsigned int* local_start=start+ptr_table[blockIdx.x];
    unsigned int* local_end=end+ptr_table[blockIdx.x];
    unsigned int* local_unq=unq+2*ptr_table[blockIdx.x];
    unsigned int elem_per_thread = (2*ctr_table[blockIdx.x]/blockDim.x)+1;
    unsigned int k_curr = tid*elem_per_thread; //Check that this makes sense
    unsigned int k_next = (tid+1)*elem_per_thread<=2*ctr_table[blockIdx.x]?(tid+1)*elem_per_thread:2*ctr_table[blockIdx.x];
    unsigned int i_curr =co_rank(local_start, local_end,ctr_table[blockIdx.x],ctr_table[blockIdx.x],k_curr);
    unsigned int i_next =co_rank(local_start, local_end,ctr_table[blockIdx.x],ctr_table[blockIdx.x],k_next);
    int j_curr = k_curr-i_curr;
    int j_next = k_next-i_next;
    merge_sequential(local_start+i_curr, local_end+j_curr, i_next-i_curr, j_next-j_curr,local_unq+k_curr);
}

__global__ void temp_Copy_Start_End(edge* edge_list, unsigned int* start, unsigned int* end, unsigned int edge_size){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    if(idx<edge_size){
        start[idx]=edge_list[idx].start;
        end[idx]=edge_list[idx].end;
    }
}

__global__ void Collect_Num_Replicas(replica_tracker* rep, unsigned int* rep_counts, unsigned int size){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    if(idx<size){
        rep_counts[idx]=rep[idx].num_replicas;
    }
}

__host__ void Org_Vertex_Helper(edge* h_edge, unsigned int* replica_count, unsigned int* h_deg, unsigned int size, unsigned int node_size){
    //Allocate memory for vertex and cluster info
    edge* d_edge;
    edge* d_edge_2;
    replica_tracker *d_tracker;
    unsigned int* d_table;
    unsigned int* d_table_2;
    unsigned int* d_table_3;

    unsigned int threads_per_block=TPB;
    unsigned int blocks_per_grid= size/threads_per_block+1;
    unsigned int blocks_per_grid_node = node_size/threads_per_block+1;
    cout<<"Num of blocks "<<blocks_per_grid<<endl;
    unsigned int ex_block_pg=(2*blocks_per_grid)/threads_per_block+1;
    cout<<"Second amount of blocks "<< ex_block_pg <<endl;
    cout<<"Allocating d_edge"<<endl;
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge, size*sizeof(edge)))){
        cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    cout<<"Copying edge list"<<endl;
    if(!HandleCUDAError(cudaMemcpy(d_edge,h_edge,size*sizeof(edge), cudaMemcpyHostToDevice))){
        cout<<"Unable to copy cluster data"<<endl;
    }
    cout<<"Done with edge list"<<endl;
    if(!HandleCUDAError(cudaMalloc((void**)&d_tracker, node_size*sizeof(replica_tracker)))){
        cout<<"Unable to allocate memory for tracker"<<endl;
    }
    // unsigned int *d_src_ptr, *d_succ;
    // if(!HandleCUDAError(cudaMalloc((void**)&d_src_ptr, node_size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for src_ptr"<<endl;
    // }
    // if(!HandleCUDAError(cudaMalloc((void**)&d_succ, size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for succ"<<endl;
    // }

    unsigned int* d_degree;
    unsigned int* d_hist;
    unsigned int* dev_fin_hist;
    unsigned int* dev_fin_count;
    unsigned int* max_val;
    unsigned int h_max_val=0;
    // unsigned int* h_hist= new unsigned int [BLOCKS*blocks_per_grid];
    
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

    if(!HandleCUDAError(cudaMalloc((void**)&d_hist, BLOCKS*blocks_per_grid*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&max_val, sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_fin_hist, BLOCKS*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&dev_fin_count, BLOCKS*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_degree, node_size*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for degree"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_degree,h_deg,node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Unable to copy degree data"<<endl;
    }
    double r = ((double) rand() / (RAND_MAX));
    cout<<"Starting random edge placement"<<endl;
    Degree_Based_Placement<<<blocks_per_grid,threads_per_block>>>(d_edge,d_degree,r,d_tracker,size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Rand_Edge Place"<<endl;
    }
    cudaFuncSetAttribute(Finalize_Replica_Tracker, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    Finalize_Replica_Tracker<<<blocks_per_grid_node,threads_per_block>>>(d_tracker,node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Finalize_Replica_Tracker"<<endl;
    }
    unsigned int* d_replica_counts; //Get the number of replicas for graphs
    if(!HandleCUDAError(cudaMalloc((void**)&d_replica_counts, node_size*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for replica counts"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_replica_counts,0,node_size*sizeof(unsigned int)))){
        cout<<"Unable to set replica counts to 0"<<endl;
    }
    Collect_Num_Replicas<<<blocks_per_grid_node,threads_per_block>>>(d_tracker,d_replica_counts,node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Collect_Num_Replicas"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(replica_count,d_replica_counts,node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy replica counts"<<endl;
    }
    Histogram_1<<<blocks_per_grid,threads_per_block>>>(d_edge,d_hist,size); 
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host with Hist_1"<<endl;
    }
    Kogge_Stone_Hist_Reduct<<<BLOCKS,blocks_per_grid, blocks_per_grid*sizeof(unsigned int)>>>(d_hist,dev_fin_hist,BLOCKS*blocks_per_grid);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host for reduce"<<endl;
    }
    Hist_Prefix_Sum<<<1,BLOCKS>>>(dev_fin_hist, dev_fin_count);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host for reduce"<<endl;
    }
    Find_Max_Cluster<<<1,BLOCKS>>>(dev_fin_hist, max_val);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Unable to synchronize with host for finding the max num of clusters"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(&h_max_val,max_val,sizeof(unsigned int), cudaMemcpyDeviceToHost))){
        cout<<"Unable to copy max val"<<endl;
    }
    HandleCUDAError(cudaFree(max_val));
    HandleCUDAError(cudaFree(d_hist));
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge_2, size*sizeof(edge)))){
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
    if(ex_block_pg>0){
        for(unsigned int i=0; i<=(unsigned int)log2(double(BLOCKS));i++){
            cout<<"Iteration "<<i<<endl;
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
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_edge_2,d_table, d_table_2,size, i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
            copy_edge_list<<<blocks_per_grid,threads_per_block>>>(d_edge,d_edge_2,size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
        }
    }
    else{
        for(unsigned int i=0; i<(unsigned int)log2(double(BLOCKS));i++){
            Sort_Cluster<<<blocks_per_grid,threads_per_block>>>(d_edge,d_table,size,i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host with Sort Cluster"<<endl;
            }
            bit_exclusive_scan<<<ex_block_pg,threads_per_block>>>(d_table,d_table_2,d_table_3,2*blocks_per_grid);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host exclusive scan"<<endl;
            }
            Swap<<<blocks_per_grid,threads_per_block>>>(d_edge,d_edge_2, d_table,d_table_2,size, i);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
            copy_edge_list<<<blocks_per_grid,threads_per_block>>>(d_edge,d_edge_2,size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Unable to synchronize with host swap"<<endl;
            }
        }
    }
    cout<<"Done with sorting"<<endl;
    HandleCUDAError(cudaFree(d_edge_2));
    HandleCUDAError(cudaFree(d_table));
    HandleCUDAError(cudaFree(d_table_2));
    HandleCUDAError(cudaFree(d_table_3));
    
    // size_t free_byte ;
    // size_t total_byte ;
    // if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
    //     cout<<"Unable to get memory info"<<endl;
    // }
    // double free_db = (double)free_byte ;
    // double total_db = (double)total_byte ;
    // double used_db = total_db - free_db ;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //     used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    // //Now, we need to organize the vertex data
    int tpb_2 = 1024;
    unsigned int *start, *end, *total;
    if(!HandleCUDAError(cudaMalloc((void**)&start, size*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for start"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&end, size*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for end"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&total, 2*size*sizeof(unsigned int)))){
        cout<<"Unable to allocate memory for unq"<<endl;
    }

    temp_Copy_Start_End<<<blocks_per_grid,threads_per_block>>>(d_edge,start,end,size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Unable to synchronize with host for temp copy start end"<<endl;
    }
    Naive_Merge_Sort<<<BLOCKS,tpb_2>>>(start,end,dev_fin_count,dev_fin_hist,total);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Unable to synchronize with host for merge sort"<<endl;
    }
    // //Now we need to merge and sort the start and end lists
    // //The edge list is sorted, let us now merge the start and end points
    // //How do we get the shared memory for the 
    // unsigned int *d_mask;
    // if(!HandleCUDAError(cudaMalloc((void**)&d_mask, 2*size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start mask"<<endl;
    // }

    // unsigned int *d_cmpt;

    // if(!HandleCUDAError(cudaMalloc((void**)&d_cmpt, size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start compt"<<endl;
    // }

    // unsigned int *d_scan;

    // if(!HandleCUDAError(cudaMalloc((void**)&d_scan, size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start scan"<<endl;
    // }

    // unsigned int *d_len;

    // if(!HandleCUDAError(cudaMalloc((void**)&d_len, BLOCKS*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start len"<<endl;
    // }

    // unsigned int* d_len_ex;

    // if(!HandleCUDAError(cudaMalloc((void**)&d_len_ex, BLOCKS*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start len"<<endl;
    // }

    // unsigned int *d_idx;

    // if(!HandleCUDAError(cudaMalloc((void**)&d_idx, 2*size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start idx"<<endl;
    // }

    // unsigned int *d_unique;
    
    // if(!HandleCUDAError(cudaMalloc((void**)&d_unique, size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for start unique"<<endl;
    // }

    // cudaFuncSetAttribute(gen_backward_mask, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    // gen_backward_mask<<<BLOCKS,tpb_2,(h_max_val)*sizeof(unsigned int)>>>(total,dev_fin_count,dev_fin_hist,d_mask,size);
    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for start mask"<<endl;
    // }


    // scan_mask<<<BLOCKS,tpb_2,(h_max_val)*sizeof(unsigned int)>>>(d_mask,d_scan,dev_fin_count,dev_fin_hist,size);

    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for start mask"<<endl;
    // }

    // if(!HandleCUDAError(cudaFree(d_mask))){
    //     cout<<"Unable to free strt mask"<<endl;
    // }
    // // //Now Perform prefix sums
    // Scanned_To_Compact<<<BLOCKS,tpb_2>>>(d_cmpt,d_scan,d_len,dev_fin_count,dev_fin_hist,size);
    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for Scanned to Compact"<<endl;
    // }

    // Final_Compression<<<BLOCKS,tpb_2>>>(d_cmpt,d_len,total,d_idx,d_unique);

    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for start mask"<<endl;
    // }
    // HandleCUDAError(cudaFree(d_cmpt));
    // HandleCUDAError(cudaFree(d_scan));

    // // /*Find the ptr values to the start and end arrays*/
    // // /*Now, we need to find the new size of the unique value array
    // // This will include performing an exclusive scan of both the end and start cluster, then 
    // // merge the two. Secondly, we need to ensure there are not replicated values in the start and end
    // // With the current implementation, this is probable. We may need to iterate through these values again*/
    // // /*Then, we can generate a hash table corresponding to the global address of the value and commence
    // // SUBLIME*/

    // unq_exclusive_scan<<<1,BLOCKS>>>(d_len,d_len_ex);

    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for start mask"<<endl;
    // }

    // if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
    //     cout<<"Unable to get memory info"<<endl;
    // }
    // free_db = (double)free_byte ;
    // total_db = (double)total_byte ;
    // used_db = total_db - free_db ;
    // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    //     used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);


    /*For the next step, this could be done quickly with a sort, then a comparision, that does not seem wise
    for this implementation*/

    /*We should make a small multithreaded kernel that goes through each of the start values and compares
    Then at the end if the end value is equal to none of them, then append it*/

    /*From this, we can then generate a local_src and local_succ*/
    //Now, we need to perform the scatter

    // if(!HandleCUDAError(cudaMemcpy(h_edge,d_edge,size*sizeof(edge),cudaMemcpyDeviceToHost))){
    //     cout<<"Unable to copy back edge data"<<endl;
    // }
    // unsigned int *d_c;
    // if(!HandleCUDAError(cudaMalloc((void**)&d_c, node_size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for c"<<endl;
    // }

    // float* d_frog_init;
    // if(!HandleCUDAError(cudaMalloc((void**)&d_frog_init, (node_size/20)*sizeof(float)))){
    //     cout<<"Unable to allocate memory for frog_init"<<endl;
    // }
    // unsigned int* d_frogs;
    // if(!HandleCUDAError(cudaMalloc((void**)&d_frogs, node_size*sizeof(unsigned int)))){
    //     cout<<"Unable to allocate memory for frogs"<<endl;
    // }
    // curandGenerator_t gen;
    // curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    // curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    // curandGenerateUniform(gen, d_frog_init, node_size/20);
    // //Figure out exec config param
    // unsigned int blocks_init_frog= (node_size/20)/TPB+1;
    // First_Init<<<blocks_init_frog,TPB>>>(d_frog_init, d_frogs, node_size, size);
    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host for first init"<<endl;
    // }
    // cudaFuncSetAttribute(FrogWild, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    // int s_mem_size=0;



    HandleCUDAError(cudaFree(d_edge));
    // HandleCUDAError(cudaFree(d_frog_init));
    // HandleCUDAError(cudaFree(d_frogs));
    // HandleCUDAError(cudaFree(d_src_ptr));
    // HandleCUDAError(cudaFree(d_succ));
    // HandleCUDAError(cudaFree(d_c));
    HandleCUDAError(cudaDeviceReset());   
}


__device__ void merge_sequential(unsigned int* start, unsigned int* end, int m, int n, unsigned int* unq){
    int i=0;
    int j=0;
    int k=0;
    while(i<m && j<n){
        if(start[i]<=end[j]){
            unq[k]=start[i];
            i++;
            k++;
        }
        else{
            unq[k]=end[j];
            j++;
            k++;
        }
    }
    while(i<m){
        unq[k]=start[i];
        i++;
        k++;
    }
    while(j<n){
        unq[k]=end[j];
        j++;
        k++;
    }
    
}


__device__ unsigned int co_rank(unsigned int* start, unsigned int* end, int m, int n, int k){
    int i = k<m ? k:m;
    int j = k-i;
    int i_low = 0>(k-n) ? 0: k-n;
    int j_low = 0>(k-m) ? 0: k-m;
    int dlt;
    bool done = false;
    while(!done){
        if(i>0 && j<n && start[i-1]>end[j]){
            dlt = (i-i_low+1)/2;
            j_low=j;
            j=j+dlt;
            i=i-dlt;
        }
        else if(j>0 && i<m && end[j-1]>=start[i]){
            dlt = (j-j_low+1)/2;
            i_low=i;
            i=i+dlt;
            j=j-dlt;
        }
        else{
            done=true;
        }
    }
    return i;
}