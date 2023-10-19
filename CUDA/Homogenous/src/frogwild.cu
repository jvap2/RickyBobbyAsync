#include "../include/data.h"
#include "../include/GPUErrors.h"

#define thrd_blck 384

__host__ void Import_Local_Src(unsigned int* local_src){
    ifstream myfile;
    myfile.open(LOCAL_SRC_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
                    else if(column>1 && column <BLOCKS+2){
                        h_replica[count-1].clusters[column-2] = stoi(word);
                    }
                    else{
                        h_replica[count-1].master_rep= stoi(word);
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
        std::cout << "Error opening file" << endl;
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
        std::cout << "Error opening file" << endl;
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
__host__ void Export_Guess(float* init_guess, unsigned int node_size){
    ofstream myfile;
    myfile.open(GUESS_PATH);
    myfile<<"vertex,values"<<endl;
    if(!myfile.is_open()){
        std::cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        for(unsigned int i=0; i<node_size; i++){
            myfile<<i<<","<<init_guess[i]<<endl;
        }
    }
}

__host__ void Export_C(unsigned int* c, unsigned int* indices, unsigned int node_size){
    ofstream myfile;
    myfile.open(C_PATH);
    myfile<<"Node,Count"<<endl;
    if(!myfile.is_open()){
        std::cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        for(unsigned int i=0; i<node_size; i++){
            myfile<<indices[i]<<","<<c[i]<<endl;
        }
    }
}

__host__ void Export_K(unsigned int* k, unsigned int node_size){
    ofstream myfile;
    myfile.open(K_PATH);
    if(!myfile.is_open()){
        std::cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        for(unsigned int i=0; i<node_size; i++){
            myfile<<k[i]<<endl;
        }
    }
}

__host__ void Capture_Top_Degree(unsigned int* degree, unsigned int* top_nodes, unsigned int node_size, unsigned int sublinear_size){
    unsigned int* nodes= new unsigned int[node_size];
    thrust::sequence(nodes, nodes+node_size,0);
    thrust::sort_by_key(thrust::host, degree, degree+node_size, nodes, thrust::greater<int>());
    thrust::copy(thrust::host, nodes, nodes+sublinear_size, top_nodes);
    delete[] nodes;
}


__host__ void FrogWild(unsigned int* local_succ, unsigned int* local_src, unsigned int* unq, unsigned int* c, unsigned int* k, unsigned int* src_ptr, 
unsigned int* unq_ptr, unsigned int* h_ptr, unsigned int* degree, unsigned int* global_src, unsigned int* global_succ,
replica_tracker* h_replica, int node_size, unsigned int edge_size, unsigned int max_unq_ctr, unsigned int version,
unsigned int* ind_rank, unsigned int debug){
    int sublinear_size=node_size/10;
    unsigned int* top_nodes= new unsigned int[sublinear_size];
    Capture_Top_Degree(degree, top_nodes, node_size, sublinear_size);
    int deviceCount=0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
                static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    int max_num_threads=0;
    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        max_num_threads=deviceProp.maxThreadsPerBlock;
    }
    unsigned int *d_succ, *d_src, *d_unq, *d_c, *d_k, *d_src_ptr, *d_unq_ptr, *d_h_ptr, *d_degree, *d_global_src, *d_global_succ;
    unsigned int *d_local_src, *d_local_succ, *d_scr_ptr;
    replica_tracker *d_replica;
    float p_t, p_s;
    unsigned int unq_mem_size=unq_ptr[BLOCKS]*sizeof(unsigned int);
    unsigned int unq_ptr_size = unq_ptr[BLOCKS];
    unsigned int unq_rand_mem_size=unq_ptr[BLOCKS]*sizeof(curandState);
    unsigned int succ_size = h_ptr[BLOCKS]*sizeof(unsigned int);
    p_s=.65;
    p_t=.15;
    unsigned int iter = 4;
    Export_Misc(iter,edge_size,BLOCKS,p_s);
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
    unsigned int *mirror_ctr;
    std::cout<<"Allocating memory for device variables"<<endl;
    if(version==0){
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq, unq_mem_size))){
            std::cout<<"Error allocating memory for d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_c, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_c"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_k, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_k"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_unq_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_replica, node_size*sizeof(replica_tracker)))){
            std::cout<<"Error allocating memory for d_replica"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_p_t, sizeof(float)))){
            std::cout<<"Error allocating memory for d_p_t"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_p_s, sizeof(float)))){
            std::cout<<"Error allocating memory for d_p_s"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&local_K, unq_mem_size))){
            std::cout<<"Error allocating memory for local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&local_C, unq_mem_size))){
            std::cout<<"Error allocating memory for local_C"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, (edge_size)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_global_succ"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&mirror_ctr, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for the mirror ctr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq, unq, unq_mem_size, cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq_ptr, unq_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_unq_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_replica, h_replica, node_size*sizeof(replica_tracker), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_replica"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_p_t, &p_t, sizeof(float), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_p_t"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_p_s, &p_s, sizeof(float), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_p_s"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_global_succ"<<endl;
        }
        float* rand_frog;
        std::cout<<"Sublinear size "<<sublinear_size<<endl;
        std::cout<<"Node size "<<node_size<<endl;
        if(!HandleCUDAError(cudaMalloc((void**)&rand_frog, sublinear_size*sizeof(float)))){
            std::cout<<"Error allocating memory for rand_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(rand_frog, 0, sublinear_size*sizeof(float)))){
            std::cout<<"Error initializing rand_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(d_k, 0, node_size*sizeof(unsigned int)))){
            std::cout<<"Error initializing d_k"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(d_c, 0, node_size*sizeof(unsigned int)))){
            std::cout<<"Error initializing d_c"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_K, 0, unq_mem_size))){
            std::cout<<"Error initializing local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C, 0, unq_mem_size))){
            std::cout<<"Error initializing local_C"<<endl;
        }
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_frog, sublinear_size);
        /*Now, we have the random numbers generated*/
        curandDestroyGenerator(gen);
        unsigned int t_per_block = thrd_blck;
        unsigned int b_per_grid_int = (sublinear_size+thrd_blck-1)/thrd_blck;
        unsigned int b_per_grid = (node_size+thrd_blck-1)/thrd_blck;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, unq_rand_mem_size))){
            std::cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, unq_rand_mem_size))){
            std::cout<<"Error allocating memory for d_state"<<endl;
        }
        std::cout<<"First init device configuration parameters"<<endl;
        std::cout<<"No. Blocks "<<b_per_grid_int<<endl;
        std::cout<<"No. Threads per block "<<t_per_block<<endl;
        First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device"<<endl;
        }
        cudaError_t err_0= cudaGetLastError();
        if (err_0 != cudaSuccess) 
            printf("First_Init Error: %s\n", cudaGetErrorString(err_0));
        cudaEvent_t start, stop;
        if(!HandleCUDAError(cudaEventCreate(&start))){
            std::cout<<"Error creating start event"<<endl;
        }
        if(!HandleCUDAError(cudaEventCreate(&stop))){
            std::cout<<"Error creating stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(start))){
            std::cout<<"Error recording start event"<<endl;
        }
        unsigned int* d_k_local_temp;
        if(!HandleCUDAError(cudaMalloc((void**)&d_k_local_temp, unq_mem_size))){
            std::cout<<"Error allocating memory for d_k_temp"<<endl;
        }
        size_t free_byte ;
        size_t total_byte ;
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            std::cout<<"Error getting memory info"<<endl;
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        std::cout<<"CUDA Dimensions"<<endl;
        std::cout<<"No. Blocks "<<BLOCKS<<endl;  
        std::cout<<"No. Threads per block "<<t_per_block<<endl;
        for(unsigned int i=0; i<iter; i++){
            std::cout<<"Iteration "<<i<<endl;
            Gather_Ver0<<<BLOCKS,thrd_blck>>>(d_k, d_unq, d_unq_ptr, local_K);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("Gather Error: %s\n", cudaGetErrorString(err));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(d_k,0, node_size*sizeof(unsigned int)))){
                std::cout<<"Error initializing d_k"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(d_k_local_temp, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
                std::cout<<"Error initializing d_k_temp"<<endl;
            }
            std::cout<<"Gathered"<<endl;
            Apply_Ver0<<<BLOCKS, thrd_blck>>>(d_unq_ptr, local_K,d_k_local_temp, local_C, d_p_t,i, d_state_teleport);
            cudaError_t err1 = cudaGetLastError();
            if (err1 != cudaSuccess) 
                printf("Apply Error: %s\n", cudaGetErrorString(err1));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device for Apply"<<endl;
            }
            std::cout<<"Applied"<<endl;
            Sync_Mirrors_Ver0<<<BLOCKS,thrd_blck>>>(d_c, d_k, d_unq, d_unq_ptr, local_C, local_K, d_global_src,d_global_succ,mirror_ctr,
            d_replica,node_size,i, d_p_s, d_state_scatter);
            cudaError_t err2 = cudaGetLastError();  
            if (err2 != cudaSuccess) 
                printf("Sync Error: %s\n", cudaGetErrorString(err2));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device for Sync"<<endl;
            }
            std::cout<<"Synced"<<endl;
            if(!HandleCUDAError(cudaMemset(mirror_ctr, 0, node_size*sizeof(unsigned int)))){
                std::cout<<"Error initializing mirror_ctr"<<endl;
            }
            std::cout<<"Scattered"<<endl;
            if(!HandleCUDAError(cudaMemset(local_K,0, unq_mem_size))){
                std::cout<<"Error initializing local_K"<<endl;
            }
        }
        Final_Commit<<<b_per_grid,thrd_blck>>>(d_c, d_k, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(stop))){
            std::cout<<"Error recording stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventSynchronize(stop))){
            std::cout<<"Error synchronizing stop event"<<endl;
        }
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout<<"Time elapsed FrogWild: "<<milliseconds<<" ms"<<endl;
        if(!HandleCUDAError(cudaMemcpy(global_src, d_global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(global_succ, d_global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to global_succ"<<endl;
        }
        
        cudaFree(d_unq);
        cudaFree(d_unq_ptr);
        cudaFree(d_replica);
        cudaFree(d_p_t);
        cudaFree(d_p_s);
        cudaFree(local_K);
        cudaFree(local_C);
        cudaFree(d_global_src);
        cudaFree(d_global_succ);
        cudaFree(d_state_teleport);
        cudaFree(d_state_scatter);
        cudaFree(rand_frog);
        thrust::sequence(ind_rank, ind_rank+node_size,1);
        unsigned int* dev_ind_ptr_approx;
        if(!HandleCUDAError(cudaMalloc((void**)&dev_ind_ptr_approx, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for dev_ind_ptr_approx"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(dev_ind_ptr_approx, ind_rank, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to dev_ind_ptr_approx"<<endl;
        }
        thrust::stable_sort_by_key(thrust::device,d_c, d_c+node_size, dev_ind_ptr_approx, thrust::greater<float>());
        if(!HandleCUDAError(cudaMemcpy(ind_rank, dev_ind_ptr_approx, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to h_indices_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(c, d_c, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to c"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(k, d_k, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to k"<<endl;
        }
        cudaFree(d_c);
        cudaFree(d_k);
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            std::cout<<"Error getting memory info"<<endl;
        }
        free_db = (double)free_byte ;
        total_db = (double)total_byte ;
        used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        //Perform PageRank with cuSparse and cuBLAS
        std::cout<<"Performing PageRank"<<endl;
        float* pagerank;
        pagerank = new float[node_size]; 
        unsigned int *indices;
        indices = new unsigned int[node_size];
        thrust::sequence(indices, indices+node_size,1);
        unsigned int max_iter = iter;
        float tol = 1e-14;   
        float damp = .15;
        float time=0.0f;
        PageRank(pagerank,indices, global_src, global_succ, damp, node_size, edge_size, max_iter, tol, &time);
        /*We need to do accuracy stuff here, for now, we need to verify with python*/

        Export_pr_vector(pagerank,indices, node_size);
        delete[] pagerank;
        delete[] indices;
    }
    else{
        float* d_export_guess;
        float* export_guess=new float[node_size];
        if(!HandleCUDAError(cudaMalloc((void**)&d_export_guess, node_size*sizeof(float)))){
            std::cout<<"Error allocating memory for d_export_guess"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_local_src,src_ptr[BLOCKS]*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_local_src"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_local_succ,succ_size))){
            std::cout<<"Error allocating memory for d_local_succ"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_scr_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_scr_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_h_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_h_ptr"<<endl;
        }  
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq, unq_mem_size))){
            std::cout<<"Error allocating memory for d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_c, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_c"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_k, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_k"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_unq_ptr, (BLOCKS+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_unq_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_replica, node_size*sizeof(replica_tracker)))){
            std::cout<<"Error allocating memory for d_replica"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_p_t, sizeof(float)))){
            std::cout<<"Error allocating memory for d_p_t"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_p_s, sizeof(float)))){
            std::cout<<"Error allocating memory for d_p_s"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&local_K, unq_mem_size))){
            std::cout<<"Error allocating memory for local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&local_C, unq_mem_size))){
            std::cout<<"Error allocating memory for local_C"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_src, (node_size+1)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&d_global_succ, (edge_size)*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_global_succ"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&mirror_ctr, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for the mirror ctr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq, unq, unq_mem_size, cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_unq"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_local_src, local_src, src_ptr[BLOCKS]*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_local_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_local_succ, local_succ, succ_size, cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_local_succ"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_scr_ptr,src_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_scr_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_h_ptr, h_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_h_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_unq_ptr, unq_ptr, (BLOCKS+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_unq_ptr"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_replica, h_replica, node_size*sizeof(replica_tracker), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_replica"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_p_t, &p_t, sizeof(float), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_p_t"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_p_s, &p_s, sizeof(float), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_p_s"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_src, global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(d_global_succ, global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to d_global_succ"<<endl;
        }
        float* rand_frog;
        std::cout<<"Sublinear size "<<sublinear_size<<endl;
        std::cout<<"Node size "<<node_size<<endl;
        if(!HandleCUDAError(cudaMalloc((void**)&rand_frog, sublinear_size*sizeof(float)))){
            std::cout<<"Error allocating memory for rand_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(rand_frog, 0, sublinear_size*sizeof(float)))){
            std::cout<<"Error initializing rand_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(d_k, 0, node_size*sizeof(unsigned int)))){
            std::cout<<"Error initializing d_k"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(d_c, 0, node_size*sizeof(unsigned int)))){
            std::cout<<"Error initializing d_c"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_K, 0, unq_mem_size))){
            std::cout<<"Error initializing local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C, 0, unq_mem_size))){
            std::cout<<"Error initializing local_C"<<endl;
        }
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        srand(time(0));
        int rand_seed = rand();
        curandSetPseudoRandomGeneratorSeed(gen, rand_seed);
        curandGenerateUniform(gen, rand_frog, sublinear_size);
        /*Now, we have the random numbers generated*/
        curandDestroyGenerator(gen);
        unsigned int t_per_block = thrd_blck;
        unsigned int b_per_grid_int = (sublinear_size+thrd_blck-1)/thrd_blck;
        unsigned int b_per_grid = (node_size+thrd_blck-1)/thrd_blck;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, unq_rand_mem_size))){
            std::cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, unq_rand_mem_size))){
            std::cout<<"Error allocating memory for d_state"<<endl;
        }
        float* d_k_init_guess;
        if(!HandleCUDAError(cudaMalloc((void**)&d_k_init_guess, node_size*sizeof(float)))){
            std::cout<<"Error allocating memory for d_k_init_guess"<<endl;
        }
        int* idx_tracker;
        if(!HandleCUDAError(cudaMalloc((void**)&idx_tracker, unq_ptr_size*sizeof(int)))){
            std::cout<<"Error allocating memory for idx_tracker"<<endl;
        }
        float* rand_node;
        if(!HandleCUDAError(cudaMalloc((void**)&rand_node, unq_ptr_size*sizeof(float)))){
            std::cout<<"Error allocating memory for rand_node"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(idx_tracker,-1, unq_ptr_size*sizeof(int)))){
            std::cout<<"Error initializing idx_tracker"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(rand_node,1.1f,unq_ptr_size*sizeof(float)))){
            std::cout<<"Error initializing rand_node"<<endl;
        }

        std::cout<<"First init device configuration parameters"<<endl;
        std::cout<<"No. Blocks "<<b_per_grid_int<<endl;
        std::cout<<"No. Threads per block "<<t_per_block<<endl;
        // First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        // if(!HandleCUDAError(cudaDeviceSynchronize())){
        //     std::cout<<"Error synchronizing device"<<endl;
        // }
        //Degree based Initialization
        unsigned int* d_top_deg_nodes;
        if(!HandleCUDAError(cudaMalloc((void**)&d_top_deg_nodes, sublinear_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for d_top_deg_nodes"<<endl;
        }
        First_Init_Deg<<<b_per_grid_int, t_per_block>>>(d_top_deg_nodes, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device"<<endl;
        }
        Copy_Init_Vector<<<b_per_grid, t_per_block>>>(d_k, d_k_init_guess, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device"<<endl;
        }
        thrust::copy(thrust::device, d_k_init_guess, d_k_init_guess+node_size, d_export_guess);
        if(!HandleCUDAError(cudaMemcpy(export_guess, d_export_guess, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to export_guess"<<endl;
        }
        Export_Guess(export_guess, node_size);
        delete[] export_guess;
        if(!HandleCUDAError(cudaFree(d_export_guess))){
            std::cout<<"Error freeing memory for d_export_guess"<<endl;
        }
        cudaError_t err_0= cudaGetLastError();
        if (err_0 != cudaSuccess) 
            printf("First_Init Error: %s\n", cudaGetErrorString(err_0));
        cudaEvent_t start, stop;
        if(!HandleCUDAError(cudaEventCreate(&start))){
            std::cout<<"Error creating start event"<<endl;
        }
        if(!HandleCUDAError(cudaEventCreate(&stop))){
            std::cout<<"Error creating stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(start))){
            std::cout<<"Error recording start event"<<endl;
        }
        unsigned int* d_k_local_temp;
        if(!HandleCUDAError(cudaMalloc((void**)&d_k_local_temp, unq_mem_size))){
            std::cout<<"Error allocating memory for d_k_temp"<<endl;
        }
        size_t free_byte ;
        size_t total_byte ;
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            std::cout<<"Error getting memory info"<<endl;
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        std::cout<<"CUDA Dimensions"<<endl;
        std::cout<<"No. Blocks "<<BLOCKS<<endl;  
        std::cout<<"No. Threads per block "<<t_per_block<<endl;
        cout<<"Initial Gather Complete"<<endl;
        if(!HandleCUDAError(cudaMemset(local_K,0,unq_mem_size))){
            std::cout<<"Error initializing local_K"<<endl;
        }
        for(unsigned int i=0; i<iter; i++){
            std::cout<<"Iteration "<<i<<endl;
            if(i==0){
                Gather_Ver0<<<BLOCKS,thrd_blck>>>(d_k, d_unq, d_unq_ptr, local_K);
                //Take the frogs from the random initialization, and then place that many on the mirrors
            }
            // else{
            //     Gather_Ver1<<<BLOCKS,thrd_blck>>>(d_k, d_unq, d_unq_ptr, local_K);
            //     //Take the frogs on the mirrors, and accumulate them so that the global version of K is accurate
            // }
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("Gather Error: %s\n", cudaGetErrorString(err));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device"<<endl;
            }
            // if(!HandleCUDAError(cudaMemset(d_k_local_temp, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            //     std::cout<<"Error initializing d_k_temp"<<endl;
            // }
            // std::cout<<"Gathered"<<endl;
            // // Apply_Ver1<<<BLOCKS, thrd_blck>>>(d_unq_ptr,d_unq, local_K,d_k_local_temp, local_C, d_p_t,i,node_size,d_c,d_k, d_state_teleport);
            Apply_Ver2<<<b_per_grid,thrd_blck>>>(d_k,d_c,d_p_t,i,d_state_teleport,node_size);
            //Iterate through the mirrors, and with 15% probability, C(i) will get incremented and K(i) will get decremented
            cudaError_t err1 = cudaGetLastError();
            if (err1 != cudaSuccess) 
                printf("Apply Error: %s\n", cudaGetErrorString(err1));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device for Apply"<<endl;
            }
            // std::cout<<"Applied"<<endl;
            Sync_Mirrors_Ver1<<<BLOCKS,thrd_blck>>>(d_c, d_k, d_unq, d_unq_ptr,d_local_src,d_local_succ,d_h_ptr,d_scr_ptr, local_C, local_K, 
            d_global_src,d_global_succ,mirror_ctr,d_replica,node_size,i, d_p_s, d_state_scatter,rand_node,idx_tracker);
            // //Decide as to which mirrors are synchronized with the global version of the graph, and then send frogs to the edges of the mirrors
            cudaError_t err2 = cudaGetLastError();  
            if (err2 != cudaSuccess) 
                printf("Sync Error: %s\n", cudaGetErrorString(err2));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device for Sync"<<endl;
            }
            // std::cout<<"Synced"<<endl;
            Scatter_Ver1<<<BLOCKS,thrd_blck>>>(d_c,d_k,d_unq,d_unq_ptr,d_local_src,d_local_succ,d_h_ptr,
            d_scr_ptr,mirror_ctr,local_K,rand_node,idx_tracker,d_p_s,node_size,i,d_state_scatter);
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                std::cout<<"Error synchronizing device for Scatter"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(mirror_ctr, 0, node_size*sizeof(unsigned int)))){
                std::cout<<"Error initializing mirror_ctr"<<endl;
            }
            // std::cout<<"Scattered"<<endl;
            // // if(!HandleCUDAError(cudaMemset(d_k,0,node_size*sizeof(unsigned int)))){
            // //     std::cout<<"Error initializing d_k"<<endl;
            // // }
            if(!HandleCUDAError(cudaMemset(idx_tracker,-1, unq_ptr_size*sizeof(int)))){
                std::cout<<"Error initializing idx_tracker"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(rand_node,1.1f,unq_ptr_size*sizeof(float)))){
                std::cout<<"Error initializing rand_node"<<endl;
            }

        }
        Final_Commit<<<b_per_grid,thrd_blck>>>(d_c, d_k, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Error synchronizing device"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(stop))){
            std::cout<<"Error recording stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventSynchronize(stop))){
            std::cout<<"Error synchronizing stop event"<<endl;
        }
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout<<"Time elapsed FrogWild: "<<milliseconds<<" ms"<<endl;
        if(!HandleCUDAError(cudaMemcpy(global_src, d_global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(global_succ, d_global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to global_succ"<<endl;
        }
        
        cudaFree(d_unq);
        cudaFree(d_unq_ptr);
        cudaFree(d_replica);
        cudaFree(d_p_t);
        cudaFree(d_p_s);
        cudaFree(local_K);
        cudaFree(local_C);
        cudaFree(d_global_src);
        cudaFree(d_global_succ);
        cudaFree(d_local_src);
        cudaFree(d_local_succ);
        cudaFree(d_h_ptr);
        cudaFree(d_state_teleport);
        cudaFree(d_state_scatter);
        cudaFree(rand_frog);
        cudaFree(rand_node);
        cudaFree(idx_tracker);
        thrust::sequence(ind_rank, ind_rank+node_size,0);
        unsigned int* dev_ind_ptr_approx;
        if(!HandleCUDAError(cudaMalloc((void**)&dev_ind_ptr_approx, node_size*sizeof(unsigned int)))){
            std::cout<<"Error allocating memory for dev_ind_ptr_approx"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(dev_ind_ptr_approx, ind_rank, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            std::cout<<"Error copying memory to dev_ind_ptr_approx"<<endl;
        }
        thrust::stable_sort_by_key(thrust::device,d_c, d_c+node_size, dev_ind_ptr_approx, thrust::greater<float>());
        if(!HandleCUDAError(cudaMemcpy(ind_rank, dev_ind_ptr_approx, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to h_indices_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(c, d_c, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to c"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(k, d_k, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to k"<<endl;
        }
        cudaFree(d_c);
        cudaFree(d_k);
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            std::cout<<"Error getting memory info"<<endl;
        }
        free_db = (double)free_byte ;
        total_db = (double)total_byte ;
        used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        //Perform PageRank with cuSparse and cuBLAS
        std::cout<<"Performing PageRank"<<endl;
        float* pagerank;
        pagerank = new float[node_size]; 
        unsigned int *indices;
        if(!HandleCUDAError(cudaMemcpy(pagerank, d_k_init_guess, node_size*sizeof(float), cudaMemcpyDeviceToHost))){
            std::cout<<"Error copying memory to pagerank"<<endl;
        }
        cudaFree(d_k_init_guess);
        indices = new unsigned int[node_size];
        thrust::sequence(indices, indices+node_size,0);
        unsigned int max_iter = 2;
        float tol = 1e-14;   
        float damp = p_t;
        float cub_time=0.0f;
        PageRank(pagerank,indices, global_src, global_succ, damp, node_size, edge_size, max_iter, tol, &cub_time);
        /*We need to do accuracy stuff here, for now, we need to verify with python*/
        Collect_Exec_Times(cub_time, milliseconds,iter,BLOCKS,node_size,p_s);

        Export_pr_vector(pagerank,indices, node_size);
        delete[] pagerank;
        delete[] indices;
    }
    delete[] top_nodes;
}

__global__ void Copy_Init_Vector(unsigned int* k, float* k_init_guess, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<node_size)
        k_init_guess[idx]=(float)k[idx];
}

__host__ void Export_Tol(float tol){
    std::ofstream myfile;
    myfile.open(TOL_PATH);
    myfile<<tol;
    myfile.close();
}

__global__ void First_Init_Deg(unsigned int* top_nodes, unsigned int* K, unsigned int node_size, unsigned int sublinear_size){
    unsigned int idx=threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<sublinear_size){
        if(top_nodes[idx]<node_size && top_nodes[idx]>=0)
            atomicAdd(&K[top_nodes[idx]],1);
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

__global__ void Gather_Ver0(unsigned int* K, unsigned int* unq, unsigned int* unq_ptr,unsigned int* local_K){
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x]; 
        //unq contains the unqiue nodes in the cluster
        //unq_ptr contains the pointers to the start of each cluster
        //Hence referencing unq[i+unq_ptr[blockIdx.x]] will give the node in the cluster, pointing to K
        //This is the node that we are going to be looking at
        if(K[unq[index]]>0){
            atomicAdd(local_K+index,K[unq[index]]);
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


__global__ void Apply_Ver0(unsigned int* unq_ptr, unsigned int* local_K_global,unsigned int* local_K_temp, unsigned int* local_C_global, float* p_t, unsigned int iter, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K_temp[i+unq_ptr[blockIdx.x]]=local_K_global[i+unq_ptr[blockIdx.x]];
    }
    __syncthreads();
    for(unsigned int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //This loop iterates throught the unique vertex values in a block
        unsigned int index=i+unq_ptr[blockIdx.x]; 
        for(int j=0; j<local_K_global[index]; j++){
            //This loop iterates through the number of living frogs on a vertex
            curand_init(1234+j+iter, idx, 0, &d_state[index]);
            float rand = curand_uniform(&d_state[index]);
            //The above section is to generate a random number for each frog
            //The index doing this seems as if it will have the same random
            //number for each frog, so incrementing the seed by j should (in theory)
            //give each frog a unique random number
            if(rand<*(p_t)){
                atomicAdd(local_C_global+index,1);
                //Increment the number of frogs which have died on this vertex-this will mirror the indexing of the unq ptr
                // atomicAdd(num_loc_C+blockIdx.x,1);
                // //Increment the number of non zero C values
                // *(local_C_idx+unq_ptr[blockIdx.x]+*(num_loc_C+blockIdx.x))=*(local_K_idx+unq_ptr[blockIdx.x]+tid);
                //The local C index in this block is going to be the same as the local K index
                //Notice that we are using the number of non-zero C's for this
                //The issue with the above part could exceed the values, this poses an issue- do we need this?
                //I do not think so
                atomicSub(local_K_temp+index,1);
                //Decrement the K value
            }
        }
    }
    __syncthreads();

    // if(tid==0){
    //     printf("BLock %d is done with iterating\n",blockIdx.x);
    // }
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K_global[i+unq_ptr[blockIdx.x]]=local_K_temp[i+unq_ptr[blockIdx.x]];
    }

    //This tells which vertices have frogs that have stopped
    // __syncthreads();
}

__global__ void Sync_Mirrors_Ver0(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_C, unsigned int* local_K, 
unsigned int* src, unsigned int* succ, unsigned int* mirror_ctr,replica_tracker* d_rep, unsigned int node_size, unsigned int iter, float* p_s, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x];   
        curand_init(1234+iter, idx, 0, &d_state[index]);
        float rand = curand_uniform(&d_state[index]);
        if(rand<*(p_s)){
            atomicAdd(mirror_ctr+unq[index],1);
            for(int j=0; j<local_C[index]; j++){
                //Commit to global memory
                    atomicAdd(C+unq[index],1);
            }
        }
    }
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x];   
        float rand = curand_uniform(&d_state[index]);
        if(K[unq[index]]>0 && rand<*(p_s)){
            unsigned int num_frog=(mirror_ctr[unq[index]]>0)?(K[unq[index]]/(mirror_ctr[unq[index]])+1):(0);
            // printf("Im going to catch %u frogs\n",num_frog);
            // printf("I am vertex %u\n",i);
            // printf("I have %u replicas\n",d_rep[i].num_replicas);
            for(int j=src[unq[index]]; j<src[unq[index]+1]; j++){
                atomicAdd(&K[succ[j]],num_frog);//Check out src and succ
            }
        }
    }
}

__global__ void Scatter_Ver0(unsigned int* C, unsigned int* K, unsigned int* src, unsigned int* succ,replica_tracker* d_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    for(int i=idx; i<node_size; i+=gridDim.x*blockDim.x){
        if(K[i]>0){
            unsigned int num_frog=K[i]/d_rep[i].num_replicas+1;
            // printf("Im going to catch %u frogs\n",num_frog);
            // printf("I am vertex %u\n",i);
            // printf("I have %u replicas\n",d_rep[i].num_replicas);
            for(int j=src[i]; j<src[i+1]; j++){
                atomicAdd(&K[succ[j]],num_frog);
                // K[i]-=(K[i]>num_frog)?(num_frog):(K[i]);
            }
        }
    }
}

__global__ void Final_Commit(unsigned int* C, unsigned int* K, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    for(int i=idx; i<node_size; i+=gridDim.x*blockDim.x){
        C[i]+=K[i];
    }
    __syncthreads();
}

__global__ void Reverse_Gather(unsigned int* K, unsigned int* local_K, replica_tracker* d_rep, unsigned int* unq, unsigned int* unq_ptr, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(unsigned int i =tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x];
        if(local_K[index]>0){
            unsigned int num_frog=local_K[index]/d_rep[unq[index]].num_replicas+1;
            atomicAdd(&K[unq[index]],num_frog);
        }
        __syncthreads();
    }
}

__global__ void Reverse_Gather_V1(unsigned int* K, unsigned int* local_K, unsigned int* unq, unsigned int* unq_ptr, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(unsigned int i =tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x];
        if(local_K[index]>0){
            unsigned int num_frog=local_K[index];
            atomicAdd(&K[unq[index]],num_frog);
        }
        __syncthreads();
    }
}

//Thoughts- maybe save multiple files of the number of nodes and commit to them with the C to sync with the mirrors

//There should be another better way


__global__ void Gather_Ver1(unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_K){
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x]; 
        //unq contains the unqiue nodes in the cluster
        //unq_ptr contains the pointers to the start of each cluster
        //Hence referencing unq[i+unq_ptr[blockIdx.x]] will give the node in the cluster, pointing to K
        //This is the node that we are going to be looking at
        if(local_K[index]>0){
            atomicAdd(K+unq[index],local_K[index]);
            //We are going to have replicas of frogs as well, additional care/attention should be made for handling this
            //Do we naively divide the count at the end by the number of replicas if there are going to be mulitplicities?
            //Possibly a question worth experimentation
            //Local_K_idx is going to store the index of the unique value with a frog on it
        }
        __syncthreads();
    }
}

//Do we send frogs within the cluster?

__global__ void Apply_Ver1(unsigned int* unq_ptr, unsigned int* unq, unsigned int* local_K_global,unsigned int* local_K_temp, unsigned int* local_C_global, float* p_t, unsigned int iter,
unsigned int node_size, unsigned int* C, unsigned int* K, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K_temp[i+unq_ptr[blockIdx.x]]=local_K_global[i+unq_ptr[blockIdx.x]];
        *(K+unq[unq_ptr[blockIdx.x]+i])=0;
    }
    __syncthreads();
    for(unsigned int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //This loop iterates throught the unique vertex values in a block
        unsigned int index=i+unq_ptr[blockIdx.x]; 
        for(int j=0; j<local_K_global[index]; j++){
            //This loop iterates through the number of living frogs on a vertex
            curand_init(1234+j+iter, 0, i, &d_state[index]);
            float rand = curand_uniform(&d_state[index]);
            //The above section is to generate a random number for each frog
            //The index doing this seems as if it will have the same random
            //number for each frog, so incrementing the seed by j should (in theory)
            //give each frog a unique random number
            if(rand<=*(p_t)){
                atomicAdd(C+unq[index],1);
                //Increment the number of frogs which have died on this vertex-this will mirror the indexing of the unq ptr
                // atomicAdd(num_loc_C+blockIdx.x,1);
                // //Increment the number of non zero C values
                // *(local_C_idx+unq_ptr[blockIdx.x]+*(num_loc_C+blockIdx.x))=*(local_K_idx+unq_ptr[blockIdx.x]+tid);
                //The local C index in this block is going to be the same as the local K index
                //Notice that we are using the number of non-zero C's for this
                //The issue with the above part could exceed the values, this poses an issue- do we need this?
                //I do not think so
                atomicSub(local_K_temp+index,1);
                //Decrement the K 
            }
        }
    }
    __syncthreads();

    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K_global[i+unq_ptr[blockIdx.x]]=local_K_temp[i+unq_ptr[blockIdx.x]];
        atomicAdd(K+unq[unq_ptr[blockIdx.x]+i],local_K_global[i+unq_ptr[blockIdx.x]]);
    }

}

__global__ void Sync_Mirrors_Ver1(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_src, unsigned int* local_succ, unsigned int* h_ptr,unsigned int* src_ptr,
unsigned int* local_C, unsigned int* local_K, unsigned int* src, unsigned int* succ, unsigned int* mirror_ctr,replica_tracker* d_rep, unsigned int node_size, unsigned int iter, float* p_s, curandState* d_state,
float* rand_node, int* idx_tracker){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x];
        unsigned int src_index = i+src_ptr[blockIdx.x];
        if(K[unq[index]]>0){
            curand_init(1234+iter+index,0, i, &d_state[index]);
            rand_node[index] = curand_uniform(&d_state[index]);
            //Synchronize the global with the distributed clusters
            //Save the vertex which will then send frogs to the next vertex in its cluster
            if(rand_node[index]<=*(p_s)){
                atomicAdd(mirror_ctr+unq[index],1);
                atomicExch(local_K+index,K[unq[index]]);
                idx_tracker[index]=src_index;
            }
            else{
                idx_tracker[index]=-1;
                rand_node[index]=1.1f;
            }
        }

    }
    __syncthreads();
}



__global__ void Scatter_Ver1(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_src, unsigned int* local_succ, unsigned int* h_ptr,unsigned int* src_ptr,
 unsigned int* mirror_ctr,unsigned int* local_K,float* rand_node, int* idx_tracker,float* p_s, unsigned int node_size, unsigned int iter, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    unsigned int num_node=len_nodes_clust/blockDim.x+1;
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        unsigned int index=i+unq_ptr[blockIdx.x]; 
        unsigned int src_index = i+src_ptr[blockIdx.x]; 
        if(rand_node[index]<=*(p_s) && mirror_ctr[unq[index]]>0){
            //Check if the random number is less than the probability of scattering, and that there are more than one mirror which is synchronized
            unsigned int num_frog=(K[unq[index]]/(mirror_ctr[unq[index]]));
            //Find the number of frogs to send based on the value of the frogs already on the vertex and the number of mirrors
            if(idx_tracker[index]>0){
                if(local_src[idx_tracker[index]+1]-local_src[idx_tracker[index]]!=0){
                    for(int j=local_src[idx_tracker[index]]; j<local_src[idx_tracker[index]+1]; j++){
                        unsigned int local_succ_val=local_succ[j+h_ptr[blockIdx.x]];//Gives the index of the local succ within
                        atomicAdd(&K[unq[local_succ_val+unq_ptr[blockIdx.x]]],num_frog);
                    }
                }
                else{
                    curand_init(42+iter+index,0, i, &d_state[index]);
                    rand_node[index] = curand_uniform(&d_state[index]);
                    unsigned int new_index = (unsigned int)floorf(rand_node[index]*unq_ptr[BLOCKS]);
                    atomicAdd(&K[unq[new_index]],1);
                }
            }
        }
    }
 }


 __global__ void Apply_Ver2(unsigned int* K_global,unsigned int* C_global, float* p_t, unsigned int iter, curandState* d_state, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<node_size){
        if(K_global[idx]>=0){
            unsigned int k_count = K_global[idx];
            for(int i = 0; i< k_count;i++){
                curand_init(1234+i+iter, idx, 0, &d_state[idx]);
                float rand = curand_uniform(&d_state[idx]);
                if(rand<*(p_t)){
                    *(C_global+idx)+=1;
                    *(K_global+idx)-=1;
                }
            }
        }
    }

}