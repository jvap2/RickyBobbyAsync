#include "../include/data.h"
#include "../include/GPUErrors.h"


__host__ void Import_Local_Src(unsigned int* local_src){
    ifstream myfile;
    myfile.open(LOCAL_SRC_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else{
                        local_src[count-1] = stoi(word);
                    }
                }
            }
            column = 0;
            count++;
        }
    }
}


__host__ void Import_Local_Succ(unsigned int* local_succ){
    ifstream myfile;
    myfile.open(LOCAL_SUCC_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else{
                        local_succ[count-1] = stoi(word);
                    }
                }
            }
            column = 0;
            count++;
        }
    }
}

__host__ void Import_Unique(unsigned int* unq){
    ifstream myfile;
    myfile.open(UNQ_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else{
                        unq[count-1] = stoi(word);
                    }
                }
            }
            column = 0;
            count++;
        }
    }
}



__host__ void Import_Src_Ctr_Ptr(unsigned int* src_ctr, unsigned int* src_ptr){
    ifstream myfile;
    myfile.open(SRC_CTR_PTR_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        src_ctr[count-1] = stoi(word);
                        column++;
                    }
                    else{
                        src_ptr[count-1] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }

}


__host__ void Import_Unq_Ptr_Ctr(unsigned int* unq_ptr, unsigned int* unq_ctr){
    ifstream myfile;
    myfile.open(UNQ_CTR_PTR_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        unq_ctr[count-1] = stoi(word);
                        column++;
                    }
                    else{
                        unq_ptr[count-1] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }

}


__host__ void Import_H_Ctr_Ptr(unsigned int* h_ctr, unsigned int* h_ptr){
    ifstream myfile;
    myfile.open(H_CTR_PTR_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        h_ctr[count-1] = stoi(word);
                        column++;
                    }
                    else{
                        h_ptr[count-1] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }

}


__host__ void Import_Degree(unsigned int* deg, unsigned int node_size){
    ifstream myfile;
    myfile.open(DEG_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else{
                        deg[count-1] = stoi(word);
                    }
                }
            }
            column = 0;
            count++;
        }
    }
}

__host__ void Import_Replica_Stats(replica_tracker* h_replica, unsigned int node_size){
    ifstream myfile;
    myfile.open(REPLICA_STAT_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        h_replica[count-1].num_replicas = stoi(word);
                        column++;
                    }
                    else{
                        h_replica[count-1].clusters[column-2] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }
}


__host__ void FrogWild(unsigned int* local_succ, unsigned int* local_src, unsigned int* unq, unsigned int* c, unsigned int* k, unsigned int* src_ptr, 
unsigned int* unq_ptr, unsigned int* h_ptr, unsigned int* degree, replica_tracker* h_replica, int node_size, unsigned int edge_size){
    unsigned int *d_succ, *d_src, *d_unq, *d_c, *d_k, *d_src_ptr, *d_unq_ptr, *d_h_ptr, *d_degree;
    replica_tracker *d_replica;
    float p_t, p_s;
    p_s=.15;
    p_t=.15;
    unsigned int iter =10;
    float* d_p_t, *d_p_s;
    unsigned int unq_ctr_max=0;
    unsigned int src_ctr_max=0;
    unsigned int h_ctr_max=0;
    unsigned int *num_local_K;
    unsigned int *num_local_C;
    for(int i = 1; i<=BLOCKS;i++){
        if(unq_ptr[i]-unq_ptr[i-1]>unq_ctr_max){
            unq_ctr_max=unq_ptr[i]-unq_ptr[i-1];
        }
        if(h_ptr[i]-h_ptr[i-1]>h_ctr_max){
            h_ctr_max=h_ptr[i]-h_ptr[i-1];
        }
        if(src_ptr[i]-src_ptr[i-1]>src_ctr_max){
            src_ctr_max=src_ptr[i]-src_ptr[i-1];
        }
    }
    unsigned int *local_K;
    unsigned int *local_C;
    unsigned int *local_K_idx;
    unsigned int *local_C_idx;
    cout<<"Allocating memory for device variables"<<endl;
    if(!HandleCUDAError(cudaMalloc((void**)&d_succ, (h_ptr[BLOCKS])*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_src, (src_ptr[BLOCKS])*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_src"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_unq, (unq_ptr[BLOCKS])*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_unq"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_c, node_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_c"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_k, node_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_k"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_src_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_src_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_unq_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_unq_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_h_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_h_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_degree, node_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_degree"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_replica, node_size*sizeof(replica_tracker)))){
        cout<<"Error allocating memory for d_replica"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_p_t, sizeof(float)))){
        cout<<"Error allocating memory for d_p_t"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_p_s, sizeof(float)))){
        cout<<"Error allocating memory for d_p_s"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&num_local_K, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error allocating memory for num_local_K"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&num_local_C, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error allocating memory for num_local_C"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&local_K, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        cout<<"Error allocating memory for local_K"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&local_C, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        cout<<"Error allocating memory for local_C"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&local_K_idx, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        cout<<"Error allocating memory for local_K_idx"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&local_C_idx, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        cout<<"Error allocating memory for local_C_idx"<<endl;
    }
    cout<<"Copying memory to device variables"<<endl;
    if(!HandleCUDAError(cudaMemcpy(d_succ, local_succ, (h_ptr[BLOCKS])*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_succ"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_src, local_src, (src_ptr[BLOCKS])*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_src"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_unq, unq, (unq_ptr[BLOCKS])*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_unq"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_c, c, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_c"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_k, k, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_k"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_src_ptr, src_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_src_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_unq_ptr, unq_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_unq_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_h_ptr, h_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_h_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_degree, degree, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_degree"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_replica, h_replica, node_size*sizeof(replica_tracker), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_replica"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_p_t, &p_t, sizeof(float), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_p_t"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_p_s, &p_s, sizeof(float), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_p_s"<<endl;
    }
    /*Now, all of the memory has been transferred and allocated*/
    /*Generate a float vector to hold the random numbers for this first intialization*/
    float* rand_frog;
    int sublinear_size=node_size/8;
    if(!HandleCUDAError(cudaMalloc((void**)&rand_frog, sublinear_size*sizeof(float)))){
        cout<<"Error allocating memory for rand_frog"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(rand_frog, 0, sublinear_size*sizeof(float)))){
        cout<<"Error initializing rand_frog"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_k, 0, node_size*sizeof(unsigned int)))){
        cout<<"Error initializing d_k"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_c, 0, node_size*sizeof(unsigned int)))){
        cout<<"Error initializing d_c"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(num_local_C, 0, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error initializing num_local_C"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(num_local_K, 0, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error initializing num_local_K"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(local_K, 0, BLOCKS*unq_ctr_max*sizeof(unsigned int)))){
        cout<<"Error initializing local_K"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(local_C, 0, BLOCKS*src_ctr_max*sizeof(unsigned int)))){
        cout<<"Error initializing local_C"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(local_K_idx, 0, BLOCKS*unq_ctr_max*sizeof(unsigned int)))){
        cout<<"Error initializing local_K_idx"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(local_C_idx, 0, BLOCKS*src_ctr_max*sizeof(unsigned int)))){
        cout<<"Error initializing local_C_idx"<<endl;
    }
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, rand_frog, sublinear_size);
    /*Now, we have the random numbers generated*/
    curandDestroyGenerator(gen);
    unsigned int t_per_block = TPB;
    unsigned int b_per_grid_int = (sublinear_size+TPB-1)/TPB;
    First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
        cout<<"Error synchronizing device"<<endl;
    }
    for(unsigned int i=0; i<iter; i++){
        Gather<<<BLOCKS,TPB>>>(d_k, d_c, d_unq, d_unq_ptr,num_local_C, num_local_K, local_K, local_C, local_K_idx, local_C_idx);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        Apply<<<BLOCKS, TPB,unq_ctr_max*sizeof(unsigned int)>>>(d_src, d_succ, d_unq, d_src_ptr, d_h_ptr, d_unq_ptr, d_k, d_c,
        num_local_C, num_local_K, i, d_p_t);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
    }
        //You will need to change the shared memory size to be the size of the unique nodes in the cluster

}



__global__ void First_Init(float* rand_frog, unsigned int* K, unsigned int node_size, unsigned int sublinear_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<sublinear_size){
        rand_frog[idx]=floorf(rand_frog[idx]*node_size);
        if(rand_frog[idx]<node_size){
            atomicAdd(&K[(unsigned int)rand_frog[idx]],1);
        }
        else{
            atomicAdd(&K[(unsigned int)rand_frog[idx]%node_size],1);
        }
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

__global__ void Gather(unsigned int* K, unsigned int* C, unsigned int* unq, unsigned int* unq_ptr, unsigned int* num_local_C, unsigned int* num_local_K,
unsigned int* local_K, unsigned int* local_C, unsigned int* local_K_idx, unsigned int* local_C_idx){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    const unsigned int c_v_len = len_nodes_clust/blockDim.x+1;
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        if(K[unq[i+unq_ptr[blockIdx.x]]]>0){
            atomicAdd(num_local_K+blockIdx.x,1);
            atomicAdd(local_K+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x],K[unq[i+unq_ptr[blockIdx.x]]]);
            atomicAdd(local_K_idx+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x],unq[i+unq_ptr[blockIdx.x]]);
            //We are going to have replicas of frogs as well, additional care/attention should be made for handling this
            //Do we naively divide the count at the end by the number of replicas if there are going to be mulitplicities?
            //Possibly a question worth experimentation
        }
        __syncthreads();
    }
}


__global__ void Apply(unsigned int* local_src, unsigned int* local_succ, unsigned int* unq, unsigned int* src_ptr, unsigned int* succ_ptr,
unsigned int* unq_ptr, unsigned int* K, unsigned int* C, unsigned int* num_loc_C, unsigned int* num_loc_K, unsigned int iter, float* p_t){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    const unsigned int c_v_len = len_nodes_clust/blockDim.x+1;
    if(tid<*(num_loc_K+blockIdx.x)){
        for(int j=0; j<)
    }

    //This tells which vertices have frogs that have stopped
    __syncthreads();
}