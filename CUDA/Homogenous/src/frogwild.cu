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

__host__ void Import_Global_Src(unsigned int* src){
    ifstream myfile;
    myfile.open(GLOBAL_SRC_PATH);
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
                    src[count-1] = stoi(word);
                }
            }
            column = 0;
            count++;
        }
    }
}


__host__ void Import_Global_Succ(unsigned int* succ){
    ifstream myfile;
    myfile.open(GLOBAL_SUCC_PATH);
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
                    succ[count-1] = stoi(word);
                }
            }
            column = 0;
            count++;
        }
    }
}


__host__ void Export_C(unsigned int* c, unsigned int node_size){
    ofstream myfile;
    myfile.open(C_PATH);
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        for(unsigned int i=0; i<node_size; i++){
            myfile<<c[i]<<endl;
        }
    }
}

__host__ void Export_K(unsigned int* k, unsigned int node_size){
    ofstream myfile;
    myfile.open(K_PATH);
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        for(unsigned int i=0; i<node_size; i++){
            myfile<<k[i]<<endl;
        }
    }
}


__host__ void FrogWild(unsigned int* local_succ, unsigned int* local_src, unsigned int* unq, unsigned int* c, unsigned int* k, unsigned int* src_ptr, 
unsigned int* unq_ptr, unsigned int* h_ptr, unsigned int* degree, unsigned int* global_src, unsigned int* global_succ,
replica_tracker* h_replica, int node_size, unsigned int edge_size, unsigned int max_unq_ctr, unsigned int* version){
    unsigned int *d_succ, *d_src, *d_unq, *d_c, *d_k, *d_src_ptr, *d_unq_ptr, *d_h_ptr, *d_degree, *d_global_src, *d_global_succ;
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
    if(version==0){
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq, (unq_ptr[BLOCKS])*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_c, node_size*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_c"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_k, node_size*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_k"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_unq_ptr"<<endl;
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
        if(!HandleCUDAError(cudaMalloc((void**)&num_local_C, BLOCKS*sizeof(unsigned int)))){
            cout<<"Error allocating memory for num_local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&num_local_K, BLOCKS*sizeof(unsigned int)))){
            cout<<"Error allocating memory for num_local_K"<<endl;
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
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, (edge_size)*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_global_succ"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq, unq, (unq_ptr[BLOCKS])*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq_ptr, unq_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_unq_ptr"<<endl;
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
        if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_global_succ"<<endl;
        }
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
        if(!HandleCUDAError(cudaMemset(local_K, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_C"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_K_idx, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_K_idx"<<endl;
        }
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, rand_frog, sublinear_size);
        /*Now, we have the random numbers generated*/
        curandDestroyGenerator(gen);
        unsigned int t_per_block = TPB;
        unsigned int b_per_grid_int = (sublinear_size+TPB-1)/TPB;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, BLOCKS*TPB*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, BLOCKS*TPB*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        for(unsigned int i=0; i<1; i++){
            cout<<"Iteration "<<i<<endl;
            Gather_Ver0<<<BLOCKS,TPB>>>(d_k, d_unq, d_unq_ptr, num_local_K, local_K, local_K_idx);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device"<<endl;
            }
            cout<<"Gathered"<<endl;
            Apply_Ver0<<<BLOCKS, TPB, max_unq_ctr*sizeof(unsigned int)>>>(d_unq, d_unq_ptr, local_K, local_C,num_local_K,local_K_idx, d_p_t,i, d_state_teleport);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device"<<endl;
            }
            cout<<"Applied"<<endl;
            Sync_Mirrors_Ver0<<<BLOCKS,TPB>>>(d_c, d_k, d_unq, d_unq_ptr, local_C, local_K, local_C_idx, local_K_idx, num_local_C, num_local_K, d_p_s, d_state_scatter);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device"<<endl;
            }
            cout<<"Synced"<<endl;
            Scatter_Ver0<<<BLOCKS,TPB>>>(d_c, d_k, d_global_src, d_global_succ, d_replica, node_size);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device"<<endl;
            }
            cout<<"Scattered"<<endl;
            if(!HandleCUDAError(cudaMemset(num_local_K, 0, BLOCKS*sizeof(unsigned int)))){
                cout<<"Error rewriting num of local K"<<endl;
            }
        }
        Final_Commit<<<BLOCKS,TPB>>>(d_c, d_k, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(c, d_c, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to c"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(k, d_k, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to k"<<endl;
        }
        cudaFree(d_unq);
        cudaFree(d_c);
        cudaFree(d_k);
        cudaFree(d_unq_ptr);
        cudaFree(d_replica);
        cudaFree(d_p_t);
        cudaFree(d_p_s);
        cudaFree(num_local_K);
        cudaFree(num_local_C);
        cudaFree(local_K);
        cudaFree(local_C);
        cudaFree(local_K_idx);
        cudaFree(d_global_src);
        cudaFree(d_global_succ);
        cudaFree(d_state_teleport);
        cudaFree(d_state_scatter);
        cudaFree(rand_frog);
    }
    else{
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
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, (edge_size)*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_global_succ"<<endl;
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
        if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to d_global_succ"<<endl;
        }
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
        if(!HandleCUDAError(cudaMemset(local_K, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_C"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_K_idx, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_K_idx"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C_idx, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
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
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, BLOCKS*TPB*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, BLOCKS*TPB*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        cudaFree(d_succ);
        cudaFree(d_src);
        cudaFree(d_unq);
        cudaFree(d_c);
        cudaFree(d_k);
        cudaFree(d_src_ptr);
        cudaFree(d_unq_ptr);
        cudaFree(d_h_ptr);
        cudaFree(d_degree);
        cudaFree(d_replica);
        cudaFree(d_p_t);
        cudaFree(d_p_s);
        cudaFree(num_local_K);
        cudaFree(num_local_C);
        cudaFree(local_K);
        cudaFree(local_C);
        cudaFree(local_K_idx);
        cudaFree(local_C_idx);
        cudaFree(d_global_src);
        cudaFree(d_global_succ);
        cudaFree(d_state_teleport);
        cudaFree(d_state_scatter);
        cudaFree(rand_frog);
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

__global__ void Gather_Ver0(unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* num_local_K,
unsigned int* local_K, unsigned int* local_K_idx){
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    const unsigned int c_v_len = len_nodes_clust/blockDim.x+1;
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //unq contains the unqiue nodes in the cluster
        //unq_ptr contains the pointers to the start of each cluster
        //Hence referencing unq[i+unq_ptr[blockIdx.x]] will give the node in the cluster, pointing to K
        //This is the node that we are going to be looking at
        if(K[unq[i+unq_ptr[blockIdx.x]]]>0){
            atomicAdd(num_local_K+blockIdx.x,1);
            atomicExch(local_K+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x]-1,K[unq[i+unq_ptr[blockIdx.x]]]);
            atomicExch(local_K_idx+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x]-1,i);
            //We are going to have replicas of frogs as well, additional care/attention should be made for handling this
            //Do we naively divide the count at the end by the number of replicas if there are going to be mulitplicities?
            //Possibly a question worth experimentation
            //Local_K_idx is going to store the index of the unique value with a frog on it
        }
        __syncthreads();
    }
    /*To summarize what has been done here
    (1), we increment the value of the number of non zero k values, i.e. we are using an array to identify how many vertices should be active in the
    next function so as to avoid warp divergence
    (2) As we increment the number of non zero K values, we use the new value as a memory pointer to identify that we need to place a new
    value in the next memory location
    (2a) Using num_local_K, we then store the K value in local_K, pointing the the block and then the offset based on the current num_local_K
    (2b) We then save the global address of K in local_K_idx*/
}


__global__ void Apply_Ver0(unsigned int* unq, unsigned int* unq_ptr, unsigned int* K, unsigned int* C, unsigned int* num_loc_K, unsigned int* local_K_idx, float* p_t, unsigned int iter, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int local_K[];
    for(int i=tid; i<*(unq_ptr+blockIdx.x+1); i+=blockDim.x){
        local_K[i]=K[unq[i+unq_ptr[blockIdx.x]]];
    }
    __syncthreads();
    for(unsigned int i=tid; i<*(unq_ptr+blockIdx.x+1); i+=blockDim.x){
        for(int j=0; j<*(K+unq_ptr[blockIdx.x]+i); j++){
            curand_init(1234+j+iter, idx, 0, &d_state[idx]);
            float rand = curand_uniform(&d_state[idx]);
            //The above section is to generate a random number for each frog
            //The index doing this seems as if it will have the same random
            //number for each frog, so incrementing the seed by j should (in theory)
            //give each frog a unique random number
            if(rand<*(p_t)){
                atomicAdd(C+unq_ptr[blockIdx.x]+local_K_idx[i],1);
                //Increment the number of frogs which have died on this vertex-this will mirror the indexing of the unq ptr
                // atomicAdd(num_loc_C+blockIdx.x,1);
                // //Increment the number of non zero C values
                // *(local_C_idx+unq_ptr[blockIdx.x]+*(num_loc_C+blockIdx.x))=*(local_K_idx+unq_ptr[blockIdx.x]+tid);
                //The local C index in this block is going to be the same as the local K index
                //Notice that we are using the number of non-zero C's for this
                //The issue with the above part could exceed the values, this poses an issue- do we need this?
                //I do not think so
                atomicSub(local_K+i,1);
                //Decrement the K value
            }
        }
    }
    for(int i=tid; i<*(unq_ptr+blockIdx.x+1); i+=blockDim.x){
        K[unq[i+unq_ptr[blockIdx.x]]]=local_K[i];
    }

    //This tells which vertices have frogs that have stopped
    // __syncthreads();
}

__global__ void Sync_Mirrors_Ver0(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_C, 
unsigned int* local_K, unsigned int* local_C_idx, unsigned int* local_K_idx, unsigned int* num_local_C, unsigned int* num_local_K, float* p_s, 
curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    curand_init(1234, idx, 0, &d_state[idx]);
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    float rand = curand_uniform(&d_state[idx]);
    for(int i=tid; i<*(unq_ptr+blockIdx.x+1); i+=blockDim.x){
        for(int j=0; j<local_C[unq_ptr[blockIdx.x]+i]; j++){
            //Commit to global memory
            if(rand<*(p_s)){
                atomicAdd(C+unq[i+unq_ptr[blockIdx.x]],1);
            }
        }
        for(int m=0; m<local_K[unq_ptr[blockIdx.x]+i]; m++){
            //Commit to global memory
            if(rand<*(p_s)){
                atomicSub(K+unq[tid+unq_ptr[blockIdx.x]],1);

            }
        }
    }
}

__global__ void Scatter_Ver0(unsigned int* C, unsigned int* K, unsigned int* src, unsigned int* succ,replica_tracker* d_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    for(int i=idx; i<node_size; i+=gridDim.x*blockDim.x){
        if(K[i]>0){
            unsigned int num_frog=K[i]/d_rep[i].num_replicas;
            for(int j=src[i]; j<src[i+1]; j++){
                atomicAdd(&K[succ[j]],num_frog);
            }
        }
    }
}

__global__ void Final_Commit(unsigned int* C, unsigned int* K, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    for(int i=idx; i<node_size; i+=gridDim.x*blockDim.x){
        C[i]+=K[i];
    }
}

//Thoughts- maybe save multiple files of the number of nodes and commit to them with the C to sync with the mirrors

//There should be another better way


__global__ void Gather_Ver1(unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* num_local_K,
unsigned int* local_K, unsigned int* local_K_idx){
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    const unsigned int c_v_len = len_nodes_clust/blockDim.x+1;
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //unq contains the unqiue nodes in the cluster
        //unq_ptr contains the pointers to the start of each cluster
        //Hence referencing unq[i+unq_ptr[blockIdx.x]] will give the node in the cluster, pointing to K
        //This is the node that we are going to be looking at
        if(K[unq[i+unq_ptr[blockIdx.x]]]>0){
            atomicAdd(num_local_K+blockIdx.x,1);
            atomicExch(local_K+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x]-1,K[unq[i+unq_ptr[blockIdx.x]]]);
            atomicExch(local_K_idx+unq_ptr[blockIdx.x]+num_local_K[blockIdx.x]-1,i);
            //We are going to have replicas of frogs as well, additional care/attention should be made for handling this
            //Do we naively divide the count at the end by the number of replicas if there are going to be mulitplicities?
            //Possibly a question worth experimentation
            //Local_K_idx is going to store the index of the unique value with a frog on it
        }
        __syncthreads();
    }
    /*To summarize what has been done here
    (1), we increment the value of the number of non zero k values, i.e. we are using an array to identify how many vertices should be active in the
    next function so as to avoid warp divergence
    (2) As we increment the number of non zero K values, we use the new value as a memory pointer to identify that we need to place a new
    value in the next memory location
    (2a) Using num_local_K, we then store the K value in local_K, pointing the the block and then the offset based on the current num_local_K
    (2b) We then save the global address of K in local_K_idx*/
}


__global__ void Apply_Ver1(unsigned int* unq, unsigned int* unq_ptr, unsigned int* K, unsigned int* C, unsigned int* num_loc_K, unsigned int* local_K_idx, float* p_t, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(tid<*(num_loc_K+blockIdx.x)){
        for(int j=0; j<*(K+unq_ptr[blockIdx.x]+tid); j++){
            curand_init(1234+j, idx, 0, &d_state[idx]);
            float rand = curand_uniform(&d_state[idx]);
            //The above section is to generate a random number for each frog
            //The index doing this seems as if it will have the same random
            //number for each frog, so incrementing the seed by j should (in theory)
            //give each frog a unique random number
            if(rand<*(p_t)){
                atomicAdd(C+unq_ptr[blockIdx.x]+local_K_idx[tid],1);
                //Increment the number of frogs which have died on this vertex-this will mirror the indexing of the unq ptr
                // atomicAdd(num_loc_C+blockIdx.x,1);
                // //Increment the number of non zero C values
                // *(local_C_idx+unq_ptr[blockIdx.x]+*(num_loc_C+blockIdx.x))=*(local_K_idx+unq_ptr[blockIdx.x]+tid);
                //The local C index in this block is going to be the same as the local K index
                //Notice that we are using the number of non-zero C's for this
                //The issue with the above part could exceed the values, this poses an issue- do we need this?
                //I do not think so
                atomicSub(K+unq_ptr[blockIdx.x]+tid,1);
                //Decrement the K value
            }
        }
    }

    //This tells which vertices have frogs that have stopped
    __syncthreads();
}

__global__ void Sync_Mirrors_Ver1(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_C, 
unsigned int* local_K, unsigned int* local_C_idx, unsigned int* local_K_idx, unsigned int* num_local_C, unsigned int* num_local_K, float* p_s, 
replica_tracker* d_replica, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    curand_init(1234, idx, 0, &d_state[idx]);
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    float rand = curand_uniform(&d_state[idx]);
    if(tid<*(unq_ptr+blockIdx.x)){
        for(int j=0; j<*(local_C+unq_ptr[blockIdx.x]+tid); j++){
            //Commit to global memory
            if(rand<*(p_s) && *(local_C+unq_ptr[blockIdx.x]+tid)>0){
                atomicAdd(C+unq[tid+unq_ptr[blockIdx.x]],1);
                *(local_C+unq_ptr[blockIdx.x]+tid)-=1;
            }
        }
        for(int m=0; m<*(local_K+unq_ptr[blockIdx.x]+tid); m++){
            //Commit to global memory
            if(rand<*(p_s) && *(local_K+unq_ptr[blockIdx.x]+tid)>0){
                atomicAdd(K+unq[tid+unq_ptr[blockIdx.x]],1);
                *(local_K+unq_ptr[blockIdx.x]+tid)-=1;
            }
        }
    }
}

__global__ void Scatter_Ver1(unsigned int* C, unsigned int* K, unsigned int* src, unsigned int* succ,replica_tracker* d_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<node_size){
        if(K[idx]>0){
            unsigned int num_frog=K[idx]/d_rep[idx].num_replicas;
            for(int i=src[idx]; i<src[idx+1]; i++){
                atomicAdd(&K[succ[i]],num_frog);
            }
        }
    }
}