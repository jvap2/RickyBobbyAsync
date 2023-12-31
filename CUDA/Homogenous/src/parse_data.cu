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

__host__ void Check_Out_Renum_Edge(edge* edge_list, int size){
    ofstream myfile;
    myfile.open(RENUM_PATH);
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

__host__ void Check_Out_Ptr_Ctr(unsigned int* h_ctr, unsigned int* h_ptr, int size){
    ofstream myfile;
    myfile.open(PTR_CTR_PATH);
    myfile <<"ptr,ctr\n";
    for(int i=0; i<size;i++){
        myfile<< to_string(h_ptr[i]);
        myfile<< ",";
        myfile<< to_string(h_ctr[i]);
        myfile<< "\n";
    }
    myfile.close();
}

__host__ void Check_Out_Unq(unsigned int* h_unq, int size){
    ofstream myfile;
    myfile.open(UNQ_PATH);
    myfile <<"unq\n";
    for(int i=0; i<size;i++){
        myfile<< to_string(h_unq[i]);
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
            std::cout<<"Rugh rogh raggy, reheheheheh"<<endl;
        }
    }
    myfile.close();
    delete[] check;
}


__host__ void check_out_replicas(string path,replica_tracker* replicas, unsigned int node_size){
    unsigned int total_rep;
    float rep_avg;
    total_rep=0;
    rep_avg=0;
    for(int i=0; i<node_size;i++){
        total_rep+=replicas[i].num_replicas;
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
    std::cout<<"Getting edge list"<<endl;
    ifstream data;
    data.open(path);
    string line,word;
    unsigned int count=0;
    unsigned int column=0;
    std::cout<<"Data is open "<<data.is_open()<<endl;
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
                        column++;
                    }
                    else{
                        arr[count-1].end=stoul(word);
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
        std::cout<<"Cannot open file"<<endl;
    }
    data.close();
}

__host__ void Check_Repeats(edge* edge_list, unsigned int size){
    for(int i=1; i<size;i++){
        if(edge_list[i].start==edge_list[i-1].start && edge_list[i].end==edge_list[i-1].end && edge_list[i].cluster==edge_list[i-1].cluster){
            std::cout<<"Repeat at "<<i<<endl;
        }
    }
}

__host__ void CSR_Graph(string path, unsigned int node_size, unsigned int edge_size, unsigned int* src_ptr, unsigned int* succ){
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
        std::cout<<"Cannot open file"<<endl;
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
    delete[] copy_ptr;
    data.close();
}

__host__ void Generate_Global_Src_Succ(unsigned int* start, unsigned int* end, unsigned int* src, unsigned int* succ, unsigned int node_size, unsigned int edge_size){
    for(int i=0; i<edge_size;i++){
        src[start[i]]++;
        succ[i]=end[i];
    }
    //Now, we need to prefix sum the src_ptr
    unsigned int* src_temp = new unsigned int[node_size+1]{0};
    for(int i=1; i<node_size;i++){
        src_temp[i]=src_temp[i-1]+src[i-1];
    }
    copy(src_temp, src_temp+node_size+1, src);
    delete[] src_temp;
}

__host__ void Sort_Edge_Start(edge* edge_list, unsigned int edge_size){
    std::stable_sort(edge_list, edge_list+edge_size, [](edge const& a, edge const& b){
        return a.start<=b.start;
    });
}


__host__ void Capture_Node_Degree(edge* edge_list, unsigned int* deg_arr, unsigned int size){
    for(unsigned int i=0; i<size;i++){
        deg_arr[edge_list[i].start]++; //Out degree
        /*deg_arr[edge_list[i].end]++;*/ //In degree
    }
}

__host__ void Export_Misc(unsigned int iterations, unsigned int edges, unsigned int blocks, float syn_pr){
    ofstream myfile;
    myfile.open(MISC_PATH);
    myfile<<"iterations,edges,blocks,p_s\n";
    myfile<<to_string(iterations);
    myfile<<",";
    myfile<<to_string(edges);
    myfile<<",";
    myfile<<to_string(blocks);
    myfile<<",";
    myfile<<to_string(syn_pr);
    myfile<<"\n";
    myfile.close();

}

__host__ void Greedy_Vertex_Cuts(edge* edgelist, replica_tracker* rep, unsigned int size){
    unsigned int* clust_mask = new unsigned int[BLOCKS]{0};
    unsigned int* cluster_count = new unsigned int[BLOCKS]{0};
    unsigned int count;
    unsigned int min_edges;
    min_edges=4294967290;
    unsigned int case_1_flag=0;
    unsigned int case_2_flag=0;
    unsigned int case_3_flag=0;
    unsigned int case_4_flag=0;
    unsigned int num_intersect = 0;
    for(int i=0; i<size;i++){   
        unsigned int start = edgelist[i].start;
        unsigned int end = edgelist[i].end;
        unsigned int start_rep = rep[start].num_replicas;
        unsigned int end_rep = rep[end].num_replicas;
        unsigned int* start_clusters = rep[start].clusters;
        unsigned int* end_clusters = rep[end].clusters;
        //Case 1, pick where the nodes intersect and place the edge there
        for(int j=0;j<BLOCKS;j++){
            if(start_clusters[j]==1 && end_clusters[j]==1){
                clust_mask[count]=j;
                count++;
                num_intersect++;
                case_1_flag=1;
            }
        }
        if(start_rep!=0 && end_rep!=0 && num_intersect==0){
            case_2_flag=1;
        }
        else if((start_rep!=0 && end_rep==0)|| (start_rep==0 && end_rep!=0)){
            case_3_flag=1;
        }
        else if(start_rep==0 && end_rep==0){
            case_4_flag=1;
        }
        if(case_1_flag){
            for(int j=0; j<num_intersect;j++){
                if(cluster_count[clust_mask[j]]<=min_edges){
                    min_edges=cluster_count[clust_mask[j]];
                    edgelist[i].cluster=clust_mask[j];
                }
            }
            rep[start].num_replicas++;
            rep[end].num_replicas++;
            rep[start].clusters[edgelist[i].cluster]|=1;
            rep[end].clusters[edgelist[i].cluster]|=1;
            cluster_count[edgelist[i].cluster]++;
            memset(clust_mask, 0, BLOCKS*sizeof(unsigned int));
        }
        //Need to check cases 2 through 4
        if(case_2_flag){
            for(int j=0; j<BLOCKS;j++){
                //Find the cluster with the least amount of between start and end
                if(rep[start].clusters[j]==1 || rep[end].clusters[j]==1){
                    if(cluster_count[j]<=min_edges){
                        min_edges=cluster_count[j];
                        edgelist[i].cluster=j;
                    }
                }
            }
            rep[start].num_replicas++;
            rep[end].num_replicas++;
            rep[start].clusters[edgelist[i].cluster]|=1;
            rep[end].clusters[edgelist[i].cluster]|=1;
            cluster_count[edgelist[i].cluster]++;
        }
        if(case_3_flag){
            if(start_rep==0){
                for(int j=0; j<BLOCKS;j++){
                    if(rep[end].clusters[j]==1){
                        if(cluster_count[j]<=min_edges){
                            min_edges=cluster_count[j];
                            edgelist[i].cluster=j;
                        }
                    }
                }
            }
            else{
                for(int j=0; j<BLOCKS;j++){
                    if(rep[start].clusters[j]==1){
                        if(cluster_count[j]<=min_edges){
                            min_edges=cluster_count[j];
                            edgelist[i].cluster=j;
                        }
                    }
                }
            }
            rep[start].num_replicas++;
            rep[end].num_replicas++;
            rep[start].clusters[edgelist[i].cluster]|=1;
            rep[end].clusters[edgelist[i].cluster]|=1;
            cluster_count[edgelist[i].cluster]++;
        }
        if(case_4_flag){
            for(int j=0; j<BLOCKS;j++){
                if(cluster_count[j]<=min_edges){
                    min_edges=cluster_count[j];
                    edgelist[i].cluster=j;
                }
            }
            rep[start].num_replicas++;
            rep[end].num_replicas++;
            rep[start].clusters[edgelist[i].cluster]=1;
            rep[end].clusters[edgelist[i].cluster]=1;
            cluster_count[edgelist[i].cluster]++;
        }
        case_1_flag=0;
        case_2_flag=0;
        case_3_flag=0;
        case_4_flag=0;
        count=0;
        num_intersect=0;
        min_edges=*max_element(cluster_count, cluster_count+BLOCKS);
    }
    for(int i=0; i<BLOCKS;i++){
        cout<<cluster_count[i]<<endl;
    }
    delete[] clust_mask;
    delete[] cluster_count;
}

__host__ void Collect_Exec_Times(float cub_time, float frog_time, unsigned int iterations, unsigned int clusters, unsigned int nodes, float p_s){
    ofstream myfile;
    myfile.open(EXEC_TIME_PATH, ios::app);
    myfile<<"\n";
    myfile<<to_string(iterations);
    myfile<< ",";
    myfile<< to_string(clusters);
    myfile<< ",";
    myfile<< to_string(nodes);
    myfile<< ",";
    myfile<< to_string(p_s);
    myfile<< ",";
    myfile<< to_string(cub_time);
    myfile<< ",";
    myfile<< to_string(frog_time);
    myfile<< ",";
    //Make room to save accuracy
    myfile<< to_string(0);
    myfile<< ",";
    myfile<< to_string(0);
    myfile<< ",";
    myfile<< to_string(0);
    myfile<< "\n";
    myfile.close();
}

__host__ void get_graph_info(string path, unsigned int* nodes, unsigned int* edges){
    std::cout<<"Getting graph info"<<endl;
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


__host__ void Export_Local_Src(unsigned int* local_src, unsigned int* h_ptr, unsigned int* h_ctr){
    ofstream myfile;
    myfile.open(LOCAL_SRC_PATH);
    myfile <<"cluster,src\n";
    for(int i = 0; i<BLOCKS; i++){
        for(int j = h_ptr[i]; j<h_ptr[i]+h_ctr[i];j++){
            myfile<< to_string(i);
            myfile<< ",";
            myfile<< to_string(local_src[j]);
            myfile<< "\n";
        }
    }
}

__host__ void Export_Local_Succ(unsigned int* local_succ, unsigned int* h_ptr, unsigned int* h_ctr){
    ofstream myfile;
    myfile.open(LOCAL_SUCC_PATH);
    myfile<<"cluster,succ\n";
    for(int i = 0 ; i< BLOCKS; i++){
        for(int j = h_ptr[i]; j<h_ptr[i]+h_ctr[i];j++){
            myfile<< to_string(i);
            myfile<< ",";
            myfile<< to_string(local_succ[j]);
            myfile<< "\n";
        }
    }
    myfile.close();
}

__host__ void Export_Global_Src(unsigned int* src, unsigned int nodes){
    ofstream myfile;
    myfile.open(GLOBAL_SRC_PATH);
    myfile<<"node,src\n";
    for(int i=0; i<=nodes;i++){
        myfile<< to_string(i);
        myfile<< ",";
        myfile<< to_string(src[i]);
        myfile<< "\n";
    }
    myfile.close();
}

__host__ void Export_Global_Succ(unsigned int* succ, unsigned int edges){
    ofstream myfile;
    myfile.open(GLOBAL_SUCC_PATH);
    myfile<<"edge,succ\n";
    for(int i=0; i<edges;i++){
        myfile<< to_string(i);
        myfile<< ",";
        myfile<< to_string(succ[i]);
        myfile<< "\n";
    }
    myfile.close();
}

__host__ void Export_Unq(unsigned int* unq, unsigned int* h_unq_ptr, unsigned int* h_unq_ctr){
    ofstream myfile;
    myfile.open(UNQ_PATH);
    myfile<<"cluster,unq\n";
    for(int i = 0; i<BLOCKS; i++){
        for(int j = h_unq_ptr[i]; j<h_unq_ptr[i]+h_unq_ctr[i];j++){
            myfile<< to_string(i);
            myfile<< ",";
            myfile<< to_string(unq[j]);
            myfile<< "\n";
        }
    }
    myfile.close();
}

__host__ void Export_Unq_Ctr_Ptr(unsigned int* h_unq_ptr, unsigned int* h_unq_ctr){
    ofstream myfile;
    myfile.open(UNQ_CTR_PTR_PATH);
    myfile<<"cluster,unq_ctr,unq_ptr\n";
    for(int i = 0; i<BLOCKS; i++){
        myfile<< to_string(i);
        myfile<< ",";
        myfile<< to_string(h_unq_ctr[i]);
        myfile<< ",";
        myfile<< to_string(h_unq_ptr[i]);
        myfile<< "\n";
    }
    myfile<< to_string(BLOCKS);
    myfile<< ",";
    myfile<< to_string(0);
    myfile<< ",";
    myfile<< to_string(h_unq_ptr[BLOCKS]);
    myfile.close();
}

__host__ void Export_Src_Ctr_Ptr(unsigned int* src_ptr, unsigned int* src_ctr){
    ofstream myfile;
    myfile.open(SRC_CTR_PTR_PATH);
    myfile<<"cluster,src_ctr,src_ptr\n";
    for(int i=0; i<BLOCKS; i++){
        myfile<< to_string(i);
        myfile<< ",";
        myfile<< to_string(src_ctr[i]);
        myfile<< ",";
        myfile<< to_string(src_ptr[i]);
        myfile<< "\n";
    }
    myfile.close();
}

__host__ void Export_H_Ctr_Ptr(unsigned int* h_ptr, unsigned int* h_ctr){
    ofstream myfile;
    myfile.open(H_CTR_PTR_PATH);
    myfile<<"cluster,h_ctr,h_ptr\n";
    for(int i=0; i<BLOCKS;i++){
        myfile<< to_string(i);
        myfile<< ",";
        myfile<< to_string(h_ctr[i]);
        myfile<< ",";
        myfile<< to_string(h_ptr[i]);
        myfile<< "\n";
    }

}

__host__ void Export_Degree(unsigned int* deg, unsigned int node_size){
    ofstream myfile;
    myfile.open(DEG_PATH);
    myfile<<"node,deg\n";
    for(int i=0; i<node_size;i++){
        myfile<< to_string(i);
        myfile<<",";
        myfile<< to_string(deg[i]);
        myfile<<"\n";
    }
    myfile.close();
}


__host__ void Export_Replica_Stats(replica_tracker* h_replica, unsigned int node_size){
    ofstream myfile;
    myfile.open(REPLICA_STAT_PATH);
    myfile<<"node,num_replicas,";
    for(int i=0; i<BLOCKS;i++){
        myfile<<to_string(i);
        myfile<<",";
    }
    myfile<<"master_rep";
    myfile<<"\n";
    for(int i=0; i<node_size;i++){
        myfile<< to_string(i);
        myfile<<",";
        myfile<< to_string(h_replica[i].num_replicas);
        myfile<<",";
        for(int j=0; j<BLOCKS;j++){
            myfile<< to_string(h_replica[i].clusters[j]);
            myfile<<",";
        }
        myfile<<to_string(h_replica[i].master_rep);
        myfile<<"\n";
    }
    myfile.close();
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


__host__ void Gen_Local_Src_Succ(edge* edge_list, unsigned int* src,unsigned int* temp_src, unsigned int* succ, unsigned int* src_ptr, unsigned int* unq, unsigned int* h_unq_ctr, unsigned int* h_unq_ptr,
unsigned int* h_ctr, unsigned int* h_ptr){
    for(int i = 0; i<BLOCKS; i++){
        //Iterate through the edges in each cluster
        for(int j=0; j<h_ctr[i];j++){
            unsigned int start = edge_list[h_ptr[i]+j].start;
            unsigned int end = edge_list[h_ptr[i]+j].end;
            if(start>=h_unq_ctr[i] || start<0 || end>=h_unq_ctr[i] || end<0){
                std::cout<<"Error: "<<start<<", "<<h_unq_ctr[i]<<endl;
                std::cout<<"Error: "<<end<<", "<<h_unq_ctr[i]<<endl;
                return;
            }
            else{
                //Increment the src_ptr
                src[src_ptr[i]+start]++;
                succ[h_ptr[i]+j]=edge_list[h_ptr[i]+j].end;
            }
        }
    }
    std::cout<<"Done with histogram and succ"<<endl;
    //Now, we need to prefix sum the src_ptr
    for(int i=0; i<BLOCKS; i++){
        temp_src[src_ptr[i]]=0;
        for(int j=src_ptr[i]+1; j<src_ptr[i+1];j++){
            temp_src[j]=temp_src[j-1]+src[j-1];
        }
    }
    std::cout<<"Done with prefix sum"<<endl;
    //Now, we need to copy the data back to src_ptr
    for(int i=0; i<BLOCKS; i++){
        for(int j=src_ptr[i]; j<src_ptr[i+1];j++){
            src[j]=temp_src[j];
        }
    }
    std::cout<<"Done with copy"<<endl;
    //Now, we need to populate the succ array
    
}



__host__ void Generate_Renum_Edgelists(edge* edge_list, edge* edge_list_2, unsigned int* unq, unsigned int* h_ptr, unsigned int* h_ctr, unsigned int* h_unq_ctr, unsigned int* h_unq_ptr){
    for(int i = 0; i<BLOCKS; i++){
        //Point to the start of the edge list
        //iterate through the starts
        for(int j=0; j<h_ctr[i];j++){
            unsigned int start = edge_list[h_ptr[i]+j].start;// Get the starting node from edge j in cluster i
            unsigned int end = edge_list[h_ptr[i]+j].end;// Get the end node from edge j in cluster i
            int start_idx = find(unq+h_unq_ptr[i], unq+h_unq_ptr[i]+h_unq_ctr[i], start)-(unq+h_unq_ptr[i]); //Return an iterator of the index, and subtract by the start position
            int end_idx = find(unq+h_unq_ptr[i], unq+h_unq_ptr[i]+h_unq_ctr[i], end)-(unq+h_unq_ptr[i]);
            if(start_idx>=h_unq_ctr[i] || end_idx>=h_unq_ctr[i] || start_idx<0 || end_idx<0){
                std::cout<<"Error: "<<start_idx<<", "<<end_idx<<", "<<h_unq_ctr[i]<<endl;
                return;
            }
            else{
                edge_list_2[h_ptr[i]+j].start=start_idx;
                edge_list_2[h_ptr[i]+j].end=end_idx;
                edge_list_2[h_ptr[i]+j].cluster=edge_list[h_ptr[i]+j].cluster;
            }
        }
    }
}

__host__ void Determine_Master(unsigned int* unq_ptr, replica_tracker* rep, unsigned int node_size){
    for(int i=0; i<node_size;i++){
        unsigned int min = 4294967290;
        for(int j=0; j<rep[i].num_replicas;j++){
            if(unq_ptr[rep[i].clusters[j]+1]-unq_ptr[rep[i].clusters[j]]<min){
                rep[i].master_rep=rep[i].clusters[j];
            }
        }
    }
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
        shared_edge[tid].start=edgelist[idx].start;
        shared_edge[tid].end=edgelist[idx].end;
        shared_edge[tid].cluster=edgelist[idx].cluster;
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
        table[blockIdx.x+gridDim.x]=ex_bits[blockDim.x-1];
    }
    __syncthreads();
    if(idx<size){
        unsigned int num_one_bef=ex_bits[tid];
        unsigned int num_one_total=ex_bits[blockDim.x];
        unsigned int dst =(bit==0)?(tid-num_one_bef):(blockDim.x-num_one_total+num_one_bef);
        // unsigned int dst = (1-bit)*(tid - num_one_bef)+ bit*(blockDim.x-num_one_total+num_one_bef);
        shared_edge[dst].cluster=key;
        shared_edge[dst].start=from;
        shared_edge[dst].end=to;
    }
    __syncthreads();
    if(idx<size){
        edgelist[idx].start=shared_edge[tid].start;
        edgelist[idx].end=shared_edge[tid].end;
        edgelist[idx].cluster=shared_edge[tid].cluster;
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
        shared_edge[tid].start=edge_list[idx].start;
        shared_edge[tid].end=edge_list[idx].end;
        shared_edge[tid].cluster=edge_list[idx].cluster;
        if(!edge_list[idx].start && !edge_list[idx].end){
            printf("Swap Error EDGELIST1: %d, %d, %d, %d, %d\n", tid, blockIdx.x, shared_edge[tid].start, shared_edge[tid].end, idx);
        }
        key = shared_edge[tid].cluster;
        bit =  (key>>iter) & 1;
    }
    __syncthreads();   
    if(idx<size){
        dst = (bit==0)? (tid+table_2[blockIdx.x]):(tid-table[blockIdx.x]+table_2[blockIdx.x+gridDim.x]);
        edge_list_2[dst].start=shared_edge[tid].start;
        edge_list_2[dst].end=shared_edge[tid].end;
        edge_list_2[dst].cluster=shared_edge[tid].cluster;
    }
    __syncthreads();
    //Check
    if(idx<size){
        if(!edge_list_2[idx].start && !edge_list_2[idx].end){
            printf("Swap Error EDGELIST2: %d, %d, %d, %d, %d\n", tid, blockIdx.x, edge_list_2[idx].start, edge_list_2[idx].end, idx);
        }
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

__global__ void final_scan_commit_scan(unsigned int* list, unsigned int* end_vals, unsigned int ptr, unsigned int size){
    unsigned int bid = blockIdx.x+ptr;
    unsigned int idx = threadIdx.x + (blockIdx.x*blockDim.x);
    if(idx<size && bid>0){
        list[idx]+=end_vals[bid-1];
    }
}


//d_table_2 contains the prefix sum
//d_table contains the counts
__global__ void copy_edge_list(edge* edge_1, edge* edge_2, unsigned int size){
    unsigned int idx=threadIdx.x+(blockDim.x*blockIdx.x);
    if(idx<size){
        edge_1[idx].start=edge_2[idx].start;
        edge_1[idx].end=edge_2[idx].end;
        edge_1[idx].cluster=edge_2[idx].cluster;
        if(!edge_1[idx].start && !edge_1[idx].end){
            printf("Copying Error: %d, %d, %d, %d, %d\n", idx, threadIdx.x, blockIdx.x, edge_1[idx].start, edge_1[idx].end);
        }
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
        // int hash = v_hash%BLOCKS;
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
        d_rep[idx].num_replicas=0;
        shared_rep[tid]=d_rep[idx];
    }
    __syncthreads();
    if(idx<node_size){
        for(int i=0; i<BLOCKS; i++){
            if(shared_rep[tid].clusters[i]==1){
                shared_rep[tid].num_replicas++;
            }
        }
    }
    __syncthreads();
    if(idx<node_size){
        d_rep[idx]=shared_rep[tid];
    }
}

__global__ void Generate_Replica_List(replica_tracker* d_rep, replica_tracker* fin_rep, unsigned int node_size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ replica_tracker shared_rep[TPB];
    __shared__ unsigned int count_rep[TPB];
    if(idx<node_size){
        shared_rep[tid]=d_rep[idx];
        count_rep[tid]=0;
    }
    __syncthreads();
    if(idx<node_size){
        for(int i=0; i<BLOCKS; i++){
            if(shared_rep[tid].clusters[i]!=0){
                fin_rep[idx].clusters[count_rep[tid]]=i;
                count_rep[tid]+=1;
                if(count_rep[tid]>BLOCKS){
                    printf("Error: %d, %d, %d\n", idx, tid, count_rep[tid]);
                }
                fin_rep[idx].num_replicas+=1;
            }
        }
    }
    __syncthreads();
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
    if(tid<BLOCKS){
        s_hist[tid]=0;
        //Initialize the histogram
    }
    if(idx<size){
        atomicAdd(s_hist+s_edge_list[tid],1);
        //Increment the histogram based on the cluster value in s_edge_list
    }
    __syncthreads();
    if(tid<BLOCKS){
        hist_bin[gridDim.x*tid+blockIdx.x]=s_hist[tid];
        //Store values for cluster x in block x
    }
    __syncthreads();
    //Now, all the data is stored locally on a blocks/grid by BLOCKS array which we need to reduce
}

__global__ void Kogge_Stone_Hist_Reduct(unsigned int* hist_bin, unsigned int* fin_bin, int size){
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ unsigned int clust_val[];
    if(idx<size){ 
        clust_val[tid]=hist_bin[idx];
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
    __syncthreads();
    if(tid==blockDim.x-1){
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



__global__ void Copy_Clusters(edge* edgelist, unsigned int* clusters, unsigned int size){
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<size){
        clusters[idx]=edgelist[idx].cluster;
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
        //Copy TPB cluster values over
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





__global__ void Find_Length_of_Unique(unsigned int* start_len, unsigned int* end_len, unsigned int* vector_length){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    __shared__ unsigned int local_size[BLOCKS];
    if(idx<BLOCKS){
        local_size[idx]=start_len[idx]+end_len[idx];
    }

}


__global__ void temp_Copy_Start_End(edge* edge_list, unsigned int* start, unsigned int* end, unsigned int edge_size){
    unsigned int idx = threadIdx.x + (blockDim.x*blockIdx.x);
    if(idx<edge_size){
        start[idx]=edge_list[idx].start;
        end[idx]=edge_list[idx].end;
    }
}


__host__ void Org_Vertex_Helper(edge* h_edge, replica_tracker* h_tracker, unsigned int* h_deg, unsigned int* h_ctr, unsigned int* h_ptr,unsigned int size, unsigned int node_size){
    //Allocate memory for vertex and cluster info
    edge* d_edge;
    edge* d_edge_2;
    replica_tracker *d_tracker;
    replica_tracker *d_tracker_fin;
    unsigned int* d_table;
    unsigned int* d_table_2;
    unsigned int* d_table_3;

    unsigned int threads_per_block=TPB;
    unsigned int blocks_per_grid= size/threads_per_block+1;
    unsigned int blocks_per_grid_node = node_size/threads_per_block+1;
    std::cout<<"Num of blocks "<<blocks_per_grid<<endl;
    unsigned int ex_block_pg=(2*blocks_per_grid)/threads_per_block+1;
    std::cout<<"Second amount of blocks "<< ex_block_pg <<endl;
    std::cout<<"Allocating d_edge"<<endl;
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge, size*sizeof(edge)))){
        std::cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    std::cout<<"Copying edge list"<<endl;
    if(!HandleCUDAError(cudaMemcpy(d_edge,h_edge,size*sizeof(edge), cudaMemcpyHostToDevice))){
        std::cout<<"Unable to copy cluster data"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**) &d_edge_2, size*sizeof(edge)))){
        std::cout<<"Unable to allocate memory for vertex data"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_edge_2,h_edge,size*sizeof(edge), cudaMemcpyHostToDevice))){
        std::cout<<"Unable to copy cluster data"<<endl;
    }
    std::cout<<"Done with edge list"<<endl;
    if(!HandleCUDAError(cudaMalloc((void**)&d_tracker, node_size*sizeof(replica_tracker)))){
        std::cout<<"Unable to allocate memory for tracker"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_tracker_fin, node_size*sizeof(replica_tracker)))){
        std::cout<<"Unable to allocate memory for tracker"<<endl;
    }

    unsigned int* d_degree;
    unsigned int* d_hist;
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
        std::cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_hist,0,BLOCKS*blocks_per_grid*sizeof(unsigned int)))){
        std::cout<<"Unable to set histogram to 0"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&max_val, sizeof(unsigned int)))){
        std::cout<<"Unable to allocate memory for histogram"<<endl;
    }
    if(!HandleCUDAError(cudaMalloc((void**)&d_degree, node_size*sizeof(unsigned int)))){
        std::cout<<"Unable to allocate memory for degree"<<endl;
    }
    if(!HandleCUDAError(cudaMemcpy(d_degree,h_deg,node_size*sizeof(unsigned int), cudaMemcpyHostToDevice))){
        std::cout<<"Unable to copy degree data"<<endl;
    }
    srand(time(0));
    int rand_seed = rand();
    double r = ( ((double)rand_seed)/(RAND_MAX));
    std::cout<<"The random number is "<<r<<endl;
    std::cout<<"Starting random edge placement"<<endl;
    Degree_Based_Placement<<<blocks_per_grid,threads_per_block>>>(d_edge,d_degree,r,d_tracker,size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Unable to synchronize with host with Rand_Edge Place"<<endl;
    }
    // Random_Edge_Placement<<<blocks_per_grid,threads_per_block>>>(d_edge,r,size);
    // if(!HandleCUDAError(cudaDeviceSynchronize())){
    //     cout<<"Unable to synchronize with host with Rand_Edge Place"<<endl;
    // }
    cudaFuncSetAttribute(Finalize_Replica_Tracker, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    Finalize_Replica_Tracker<<<blocks_per_grid_node,threads_per_block>>>(d_tracker,node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Unable to synchronize with host with Finalize_Replica_Tracker"<<endl;
    }
    cudaFuncSetAttribute(Generate_Replica_List, cudaFuncAttributeMaxDynamicSharedMemorySize, 102400);
    Generate_Replica_List<<<blocks_per_grid_node,threads_per_block>>>(d_tracker,d_tracker_fin,node_size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Unable to synchronize with host with Generate_Replica_List"<<endl;
    }    
    if(!HandleCUDAError(cudaMemcpy(h_tracker,d_tracker_fin,node_size*sizeof(replica_tracker), cudaMemcpyDeviceToHost))){
        std::cout<<"Unable to copy tracker data"<<endl;
    }
    unsigned int* d_cluster;
    if(!HandleCUDAError(cudaMalloc((void**)&d_cluster, size*sizeof(unsigned int)))){
        std::cout<<"Unable to allocate memory for cluster"<<endl;
    }
    if(!HandleCUDAError(cudaMemset(d_cluster,0,size*sizeof(unsigned int)))){
        std::cout<<"Unable to set cluster to 0"<<endl;
    }
    Copy_Clusters<<<blocks_per_grid,threads_per_block>>>(d_edge,d_cluster,size);
    if(!HandleCUDAError(cudaDeviceSynchronize())){
            std::cout<<"Unable to synchronize with host with Copy_Clusters"<<endl;
    }
    for(int i = 0; i<BLOCKS; i++){
        h_ctr[i]=thrust::count(thrust::device, d_cluster, d_cluster+size, i);
    }
    HandleCUDAError(cudaFree(max_val));
    HandleCUDAError(cudaFree(d_hist));
    thrust::exclusive_scan(thrust::host, h_ctr, h_ctr+BLOCKS, h_ptr);
    if(!HandleCUDAError(cudaMemcpy(h_edge,d_edge,size*sizeof(edge),cudaMemcpyDeviceToHost))){
        std::cout<<"Unable to copy back edge data"<<endl;
    }

    HandleCUDAError(cudaFree(d_edge));
    HandleCUDAError(cudaFree(d_degree));
    HandleCUDAError(cudaFree(d_tracker));
    HandleCUDAError(cudaFree(d_tracker_fin));
    HandleCUDAError(cudaFree(d_cluster));
    HandleCUDAError(cudaDeviceReset());   
}


__device__ __host__ void merge_sequential(unsigned int* start, unsigned int* end, int m, int n, unsigned int* unq){
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
