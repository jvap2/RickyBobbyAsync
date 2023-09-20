#include "../include/data.h"
#include "../include/GPUErrors.h"

#define thrd_blck 512

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


__host__ void Export_C(unsigned int* c, unsigned int* indices, unsigned int node_size){
    ofstream myfile;
    myfile.open(C_PATH);
    myfile<<"Node,Count"<<endl;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
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
replica_tracker* h_replica, int node_size, unsigned int edge_size, unsigned int max_unq_ctr, unsigned int version,
unsigned int* ind_rank, unsigned int debug){
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
    replica_tracker *d_replica;
    float p_t, p_s;
    p_s=.85;
    p_t=.85;
    unsigned int iter = 10;
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
        if(!HandleCUDAError(cudaMalloc((void**)&local_K, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error allocating memory for local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMalloc((void**)&local_C, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error allocating memory for local_C"<<endl;
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
        int sublinear_size=node_size/300;
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
        if(!HandleCUDAError(cudaMemset(local_K, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_K"<<endl;
        }
        if(!HandleCUDAError(cudaMemset(local_C, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error initializing local_C"<<endl;
        }
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, rand_frog, sublinear_size);
        /*Now, we have the random numbers generated*/
        curandDestroyGenerator(gen);
        unsigned int t_per_block = thrd_blck;
        unsigned int b_per_grid_int = (sublinear_size+thrd_blck-1)/thrd_blck;
        unsigned int b_per_grid = (node_size+thrd_blck-1)/thrd_blck;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, BLOCKS*thrd_blck*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, BLOCKS*thrd_blck*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        cudaEvent_t start, stop;
        if(!HandleCUDAError(cudaEventCreate(&start))){
            cout<<"Error creating start event"<<endl;
        }
        if(!HandleCUDAError(cudaEventCreate(&stop))){
            cout<<"Error creating stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(start))){
            cout<<"Error recording start event"<<endl;
        }
        // unsigned int num_streams=max_unq_ctr*sizeof(unsigned int)/102400+1;
        // cudaStream_t streams[num_streams];
        // for(int i=0; i<num_streams;i++){
        //     if(!HandleCUDAError(cudaStreamCreate(&streams[i]))){
        //         cout<<"Error creating stream"<<endl;
        //     }
        // }
        unsigned int* d_k_local_temp;
        if(!HandleCUDAError(cudaMalloc((void**)&d_k_local_temp, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
            cout<<"Error allocating memory for d_k_temp"<<endl;
        }
        size_t free_byte ;
        size_t total_byte ;
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            cout<<"Error getting memory info"<<endl;
        }
        double free_db = (double)free_byte ;
        double total_db = (double)total_byte ;
        double used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        // cudaFuncSetAttribute(Apply_Ver0, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
        for(unsigned int i=0; i<iter; i++){
            cout<<"Iteration "<<i<<endl;
            Gather_Ver0<<<BLOCKS,thrd_blck>>>(d_k, d_unq, d_unq_ptr, local_K);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) 
                printf("Gather Error: %s\n", cudaGetErrorString(err));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(d_k,0, node_size*sizeof(unsigned int)))){
                cout<<"Error initializing d_k"<<endl;
            }
            if(!HandleCUDAError(cudaMemset(d_k_local_temp, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
                cout<<"Error initializing d_k_temp"<<endl;
            }
            cout<<"Gathered"<<endl;
            cout<<max_unq_ctr*sizeof(unsigned int)<<endl;
            Apply_Ver0<<<BLOCKS, thrd_blck>>>(d_unq_ptr, local_K,d_k_local_temp, local_C, d_p_t,i, d_state_teleport);
            cudaError_t err1 = cudaGetLastError();
            if (err1 != cudaSuccess) 
                printf("Apply Error: %s\n", cudaGetErrorString(err1));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device for Apply"<<endl;
            }
            cout<<"Applied"<<endl;
            Sync_Mirrors_Ver0<<<BLOCKS,thrd_blck>>>(d_c, d_k, d_unq, d_unq_ptr, local_C, local_K, d_p_s, d_state_scatter);
            cudaError_t err2 = cudaGetLastError();  
            if (err2 != cudaSuccess) 
                printf("Sync Error: %s\n", cudaGetErrorString(err2));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device for Sync"<<endl;
            }
            cout<<"Synced"<<endl;
            Scatter_Ver0<<<b_per_grid,thrd_blck>>>(d_c, d_k, d_global_src, d_global_succ, d_replica, node_size);
            err2= cudaGetLastError();
            if (err2 != cudaSuccess) 
                printf("Error: %s\n", cudaGetErrorString(err2));
            if(!HandleCUDAError(cudaDeviceSynchronize())){
                cout<<"Error synchronizing device for Scatter"<<endl;
            }
            cout<<"Scattered"<<endl;
        }
        Final_Commit<<<b_per_grid,thrd_blck>>>(d_c, d_k, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(stop))){
            cout<<"Error recording stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventSynchronize(stop))){
            cout<<"Error synchronizing stop event"<<endl;
        }
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout<<"Time elapsed FrogWild: "<<milliseconds<<" ms"<<endl;
        if(!HandleCUDAError(cudaMemcpy(global_src, d_global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(global_succ, d_global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to global_succ"<<endl;
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
            cout<<"Error allocating memory for dev_ind_ptr_approx"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(dev_ind_ptr_approx, ind_rank, node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
            cout<<"Error copying memory to dev_ind_ptr_approx"<<endl;
        }
        thrust::stable_sort_by_key(thrust::device,d_c, d_c+node_size, dev_ind_ptr_approx, thrust::greater<float>());
        if(!HandleCUDAError(cudaMemcpy(ind_rank, dev_ind_ptr_approx, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to h_indices_frog"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(c, d_c, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to c"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(k, d_k, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to k"<<endl;
        }
        cudaFree(d_c);
        cudaFree(d_k);
        if(!HandleCUDAError(cudaMemGetInfo( &free_byte, &total_byte ))){
            cout<<"Error getting memory info"<<endl;
        }
        free_db = (double)free_byte ;
        total_db = (double)total_byte ;
        used_db = total_db - free_db ;
        printf("GPU memory usage before PR: used = %f, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
        //Perform PageRank with cuSparse and cuBLAS
        // cout<<"Performing PageRank"<<endl;
        // float* pagerank;
        // pagerank = new float[node_size]; 
        // unsigned int *indices, *indices_approx;
        // indices = new unsigned int[node_size];
        // indices_approx = new unsigned int[node_size];
        // thrust::sequence(indices, indices+node_size,1);
        // thrust::sequence(indices_approx, indices_approx+node_size,1);
        // unsigned int max_iter = 100;
        // float tol = 1e-14;   
        // float damp = p_t;
        // PageRank(pagerank,indices, global_src, global_succ, damp, node_size, edge_size, max_iter, tol);
        // /*We need to do accuracy stuff here, for now, we need to verify with python*/

        // Export_pr_vector(pagerank,indices, node_size);
        // delete[] pagerank;
        // delete[] indices;
        // delete[] indices_approx;
    }
    else if(version==1){
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
        unsigned int t_per_block = thrd_blck;
        unsigned int b_per_grid_int = (sublinear_size+thrd_blck-1)/thrd_blck;
        unsigned int b_per_grid = (node_size+thrd_blck-1)/thrd_blck;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, BLOCKS*thrd_blck*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, BLOCKS*thrd_blck*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        First_Init<<<b_per_grid_int, t_per_block>>>(rand_frog, d_k, node_size, sublinear_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        cudaEvent_t start, stop;
        if(!HandleCUDAError(cudaEventCreate(&start))){
            cout<<"Error creating start event"<<endl;
        }
        if(!HandleCUDAError(cudaEventCreate(&stop))){
            cout<<"Error creating stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventRecord(start))){
            cout<<"Error recording start event"<<endl;
        }
        cudaFuncSetAttribute(Apply_Ver0, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
        // for(unsigned int i=0; i<iter; i++){
        //     cout<<"Iteration "<<i<<endl;
        //     Gather_Ver1<<<BLOCKS,thrd_blck>>>(d_k, d_unq, d_unq_ptr, num_local_K, local_K, local_K_idx);
        //     if(!HandleCUDAError(cudaDeviceSynchronize())){
        //         cout<<"Error synchronizing device"<<endl;
        //     }
        //     cout<<"Gathered"<<endl;
        //     cout<<max_unq_ctr*sizeof(unsigned int)<<endl;
        //     Apply_Ver1<<<BLOCKS, thrd_blck, max_unq_ctr*sizeof(unsigned int)>>>(d_unq_ptr, local_K, local_C,num_local_K,local_K_idx, d_p_t,i, d_state_teleport);
        //     if(!HandleCUDAError(cudaDeviceSynchronize())){
        //         cout<<"Error synchronizing device for Apply"<<endl;
        //     }
        //     cout<<"Applied"<<endl;
        //     Sync_Mirrors_Ver1<<<BLOCKS,thrd_blck>>>(d_c, d_k, d_unq, d_unq_ptr, local_C, local_K, local_C_idx, local_K_idx, num_local_C, num_local_K, d_p_s, d_state_scatter);
        //     if(!HandleCUDAError(cudaDeviceSynchronize())){
        //         cout<<"Error synchronizing device for Sync"<<endl;
        //     }
        //     cout<<"Synced"<<endl;
        //     Scatter_Ver1<<<b_per_grid,thrd_blck>>>(d_c, d_k, d_global_src, d_global_succ, d_replica, node_size);
        //     if(!HandleCUDAError(cudaDeviceSynchronize())){
        //         cout<<"Error synchronizing device for Scatter"<<endl;
        //     }
        //     cout<<"Scattered"<<endl;
        //     if(!HandleCUDAError(cudaMemset(num_local_K, 0, BLOCKS*sizeof(unsigned int)))){
        //         cout<<"Error rewriting num of local K"<<endl;
        //     }
        //     if(!HandleCUDAError(cudaMemset(num_local_C, 0, BLOCKS*sizeof(unsigned int)))){
        //         cout<<"Error rewriting num of local C"<<endl;
        //     }
        //     if(!HandleCUDAError(cudaMemset(local_K, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        //         cout<<"Error rewriting local K"<<endl;
        //     }
        //     if(!HandleCUDAError(cudaMemset(local_C, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        //         cout<<"Error rewriting local C"<<endl;
        //     }
        //     if(!HandleCUDAError(cudaMemset(local_K_idx, 0, unq_ptr[BLOCKS]*sizeof(unsigned int)))){
        //         cout<<"Error rewriting local K idx"<<endl;
        //     }
        // }
        if(!HandleCUDAError(cudaEventRecord(stop))){
            cout<<"Error recording stop event"<<endl;
        }
        if(!HandleCUDAError(cudaEventSynchronize(stop))){
            cout<<"Error synchronizing stop event"<<endl;
        }
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout<<"Time elapsed FrogWild: "<<milliseconds<<" ms"<<endl;
        Final_Commit<<<BLOCKS,thrd_blck>>>(d_c, d_k, node_size);
        if(!HandleCUDAError(cudaDeviceSynchronize())){
            cout<<"Error synchronizing device"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(global_src, d_global_src, (node_size+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to global_src"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(global_succ, d_global_succ, (edge_size)*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to global_succ"<<endl;
        }
        
        cudaFree(d_unq);
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
        //Perform PageRank with cuSparse and cuBLAS
        cout<<"Performing PageRank"<<endl;
        float* pagerank;
        pagerank = new float[node_size]; 
        unsigned int *indices, *indices_approx;
        indices = new unsigned int[node_size];
        indices_approx = new unsigned int[node_size];
        unsigned int* dev_ind_ptr_approx;
        if(!HandleCUDAError(cudaMalloc((void**)&dev_ind_ptr_approx, node_size*sizeof(unsigned int)))){
            cout<<"Error allocating memory for dev_ind_ptr_approx"<<endl;
        }
        thrust::sequence(indices, indices+node_size,1);
        thrust::sequence(indices_approx, indices_approx+node_size,1);
        unsigned int max_iter = 100;
        float tol = 1e-14;   
        float damp = p_t;
        PageRank(pagerank,indices, global_src, global_succ, damp, node_size, edge_size, max_iter, tol);
        /*We need to do accuracy stuff here, for now, we need to verify with python*/

        Export_pr_vector(pagerank,indices, node_size);
        thrust::stable_sort_by_key(thrust::device,d_c, d_c+node_size, dev_ind_ptr_approx, thrust::greater<float>());
        if(!HandleCUDAError(cudaMemcpy(ind_rank, dev_ind_ptr_approx, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to h_indices_frog"<<endl;
        }
        delete[] pagerank;
        delete[] indices;
        delete[] indices_approx;
        if(!HandleCUDAError(cudaMemcpy(c, d_c, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to c"<<endl;
        }
        if(!HandleCUDAError(cudaMemcpy(k, d_k, node_size*sizeof(unsigned int), cudaMemcpyDeviceToHost))){
            cout<<"Error copying memory to k"<<endl;
        }
        cudaFree(d_c);
        cudaFree(d_k);

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
        unsigned int t_per_block = thrd_blck;
        unsigned int b_per_grid_int = (sublinear_size+thrd_blck-1)/thrd_blck;
        curandState* d_state_teleport;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_teleport, BLOCKS*thrd_blck*sizeof(curandState)))){
            cout<<"Error allocating memory for d_state"<<endl;
        }
        curandState* d_state_scatter;
        if(!HandleCUDAError(cudaMalloc((void**)&d_state_scatter, BLOCKS*thrd_blck*sizeof(curandState)))){
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

__global__ void Gather_Ver0(unsigned int* K, unsigned int* unq, unsigned int* unq_ptr,unsigned int* local_K){
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    const unsigned int c_v_len = len_nodes_clust/blockDim.x+1;
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //unq contains the unqiue nodes in the cluster
        //unq_ptr contains the pointers to the start of each cluster
        //Hence referencing unq[i+unq_ptr[blockIdx.x]] will give the node in the cluster, pointing to K
        //This is the node that we are going to be looking at
        if(K[unq[i+unq_ptr[blockIdx.x]]]>0){
            *(local_K+unq_ptr[blockIdx.x]+i)=K[unq[i+unq_ptr[blockIdx.x]]];
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
    unsigned int* local_K = local_K_temp+unq_ptr[blockIdx.x];
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K[i]=local_K_global[i+unq_ptr[blockIdx.x]];
    }
    __syncthreads();
    // if(tid==0){
    //     printf("BLock %d is done\n",blockIdx.x);
    // }

    for(unsigned int i=tid; i<len_nodes_clust; i+=blockDim.x){
        //This loop iterates throught the unique vertex values in a block
        for(int j=0; j<*(local_K_global+unq_ptr[blockIdx.x]+i); j++){
            //This loop iterates through the number of living frogs on a vertex
            curand_init(1234+j+iter, idx, 0, &d_state[idx]);
            float rand = curand_uniform(&d_state[idx]);
            //The above section is to generate a random number for each frog
            //The index doing this seems as if it will have the same random
            //number for each frog, so incrementing the seed by j should (in theory)
            //give each frog a unique random number
            if(rand<*(p_t)){
                atomicAdd(local_C_global+unq_ptr[blockIdx.x]+i,1);
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
        __syncthreads();
    }
    // if(tid==0){
    //     printf("BLock %d is done with iterating\n",blockIdx.x);
    // }
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        local_K_global[i+unq_ptr[blockIdx.x]]=local_K[i];
    }

    //This tells which vertices have frogs that have stopped
    // __syncthreads();
}

__global__ void Sync_Mirrors_Ver0(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_C, unsigned int* local_K, float* p_s, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    curand_init(1234, idx, 0, &d_state[idx]);
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    float rand = curand_uniform(&d_state[idx]);
    for(int i=tid; i<len_nodes_clust; i+=blockDim.x){
        for(int j=0; j<local_C[unq_ptr[blockIdx.x]+i]; j++){
            //Commit to global memory
            if(rand<*(p_s)){
                atomicAdd(C+unq[i+unq_ptr[blockIdx.x]],1);
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
        if(local_K[i]>0){
            unsigned int num_frog=local_K[i]/d_rep[unq[i+unq_ptr[blockIdx.x]]].num_replicas+1;
            for(int j=0; j<num_frog; j++){
                atomicAdd(&K[unq[i+unq_ptr[blockIdx.x]]],1);
            }
        }
        __syncthreads();
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


__global__ void Apply_Ver1(unsigned int* unq_ptr, unsigned int* K, unsigned int* C, unsigned int* num_loc_K, unsigned int* local_K_idx, float* p_t, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int local_K[];
    for(int i = tid; i<num_loc_K[blockIdx.x]; i+=blockDim.x){
        local_K[i]=K[unq_ptr[blockIdx.x]+i];
    }
    for(unsigned int i=tid; i<num_loc_K[blockIdx.x]; i+=blockDim.x){
        for(int j=0; j<*(K+unq_ptr[blockIdx.x]+i); j++){
            curand_init(1234+j, idx, 0, &d_state[idx]);
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
                if(local_K[i]==0){
                    atomicSub(num_loc_K+blockIdx.x,1);
                }
            }
        }
    }

    //This tells which vertices have frogs that have stopped
    __syncthreads();
}

__global__ void Sync_Mirrors_Ver1(unsigned int* C, unsigned int* K, unsigned int* unq, unsigned int* unq_ptr, unsigned int* local_C, 
unsigned int* local_K, float* p_s, replica_tracker* d_replica, curandState* d_state){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    const unsigned int len_nodes_clust=unq_ptr[blockIdx.x+1]-unq_ptr[blockIdx.x];
    curand_init(1234, idx, 0, &d_state[idx]);
    //We have this outside so if the if condition is satisfied, the entirety of local C can be committed
    //to the global C
    float rand = curand_uniform(&d_state[idx]);
    for(unsigned int i =tid; i<len_nodes_clust; i+=blockDim.x){
        for(int j=0; j<*(local_C+unq_ptr[blockIdx.x]+i); j++){
            //Commit to global memory
            if(rand<*(p_s) && *(local_C+unq_ptr[blockIdx.x]+i)>0){
                atomicAdd(C+unq[i+unq_ptr[blockIdx.x]],1);
                *(local_C+unq_ptr[blockIdx.x]+i)-=1;
            }
        }
        __syncthreads();
    }
}

__global__ void Scatter_Ver1(unsigned int* C, unsigned int* K, unsigned int* src, unsigned int* succ,replica_tracker* d_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(idx<node_size){
        if(K[idx]>0){
            unsigned int num_frog=K[idx]/d_rep[idx].num_replicas+1;
            for(int i=src[idx]; i<src[idx+1]; i++){
                atomicAdd(&K[succ[i]],num_frog);
            }
        }
    }
}


__global__ void Schur_Product_Vectors(unsigned int* vect_1, unsigned int* vect_2, unsigned int* res_vec, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<size){
        res_vec[idx]=vect_1[idx]*vect_2[idx];
    }
}

__global__ void Partial_Sums(unsigned int* res_vec, unsigned int* last_val, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int temp=0;
    if (idx>=size){
        return;
    }
    unsigned int* blockAddress= res_vec+(blockDim.x*blockIdx.x);
    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if (tid<stride && tid+stride<size){
            blockAddress[tid]+=blockAddress[tid+stride];
        }
        __syncthreads();
    }
    if(tid==0){
        last_val[blockIdx.x]=blockAddress[0];
    }
}

__global__ void Compute_L2_Max_u_1(unsigned int* vect_1, unsigned int* res_vec_1, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<size){
        res_vec_1[idx]=vect_1[idx]*vect_1[idx];
    }
}


__global__ void Partial_Sum_Last_Val(unsigned int* last_val, unsigned int* res, unsigned int block_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int* blockAddress=last_val;
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			//tid<stride ensures we do not try to access memory past the vector allocated to the block
			//tid+stride<size allows for vector sizes less than blockDim
			blockAddress[tid] += blockAddress[tid + stride];
		}
		__syncthreads();//Make all of the threads wait to go to the next iteration so the values are up to date
	}
    if(tid==0){
        *(res)=(blockAddress[0]);
    }

}


__host__ void Verif_Dot_Product(unsigned int* vec_1, unsigned int* vec_2, unsigned int res, unsigned int size){
    unsigned int temp=0;
    for(int i=0; i<size; i++){
        temp+=vec_1[i]*vec_2[i];
    }
    if(temp==res){
        cout<<"Dot product is correct"<<endl;
    }
    else{
        cout<<"Dot product is incorrect"<<endl;
        cout<<"GPU "<<res<<endl;
        cout<<"CPU "<<temp<<endl;
    }
}

__host__ void Verif_L2(unsigned int* vec, unsigned int res, unsigned int size){
    unsigned int temp=0;
    for(int i=0; i<size; i++){
        temp+=vec[i]*vec[i];
    }
    if(temp==res){
        cout<<"L2 norm is correct"<<endl;
    }
    else{
        cout<<"L2 norm is incorrect"<<endl;
        cout<<"GPU "<<res<<endl;
        cout<<"CPU "<<temp<<endl;
    }
} 