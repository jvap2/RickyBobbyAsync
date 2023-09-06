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
    if(!HandleCUDAError(cudaMalloc((void**)&d_src_ptr, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_src_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_unq_ptr, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_unq_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_h_ptr, BLOCKS*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_h_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_degree, node_size*sizeof(unsigned int)))){
        cout<<"Error allocating memory for d_degree"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_replica, node_size*sizeof(replica_tracker)))){
        cout<<"Error allocating memory for d_replica"<<endl;
    }
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
    if(!HandleCUDAError(cudaMemcpy(d_src_ptr, src_ptr, BLOCKS*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_src_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_unq_ptr, unq_ptr, BLOCKS*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_unq_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_h_ptr, h_ptr, BLOCKS*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_h_ptr"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_degree, degree, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_degree"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_replica, h_replica, node_size*sizeof(replica_tracker), cudaMemcpyHostToDevice))){
        cout<<"Error copying memory to d_replica"<<endl;
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

__global__ void Apply(unsigned int* local_src, unsigned int* local_succ, unsigned int* src_ptr, unsigned int* succ_ptr,
unsigned int* K, unsigned int* C,unsigned int iter, float p_t){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;

}