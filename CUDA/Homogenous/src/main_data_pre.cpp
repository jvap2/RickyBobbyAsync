#include "../include/data.h"

// #define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
// #define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    unsigned int *src, *succ;
    src = new unsigned int[nodes+1]{0};
    succ = new unsigned int[edges]{0};

    cout<<nodes<<endl;
    cout<<edges<<endl;
    edge* edge_list;
    unsigned int* deg = new unsigned int[nodes]{0};
    edge_list=(edge*)malloc(sizeof(edge)*edges);
    replica_tracker* h_replica = (replica_tracker*)malloc(sizeof(replica_tracker)*nodes);
    unsigned int* replica = new unsigned int[edges];
    cout<<"Starting the edge list function"<<endl;
    return_edge_list(EDGE_PATH,edge_list);
    cout<<"Finished the edge list function"<<endl;
    Capture_Node_Degree(edge_list,deg,edges);
    Export_Degree(deg,nodes);
    unsigned int *h_ctr, *h_ptr;
    h_ctr = new unsigned int[BLOCKS]{0};
    h_ptr = new unsigned int[BLOCKS+1]{0};
    unsigned int *h_unq;
    h_unq = new unsigned int[edges];
    cout<<"Ending edge list function"<<endl;
    cout<<"Starting Helper Function"<<endl;
    // Greedy_Vertex_Cuts(edge_list,h_replica,edges);
    unsigned int* h_start = new unsigned int[edges]{0}; //Collect all starting node values
    unsigned int* h_end = new unsigned int[edges]{0}; //collect all ending node values 
    unsigned int* h_start_global = new unsigned int[edges]{0}; //Collect all starting node values
    unsigned int* h_end_global = new unsigned int[edges]{0}; //collect all ending node values
    unsigned int* h_cluster = new unsigned int[edges]{0};
    unsigned int* h_start_global_2= new unsigned int[edges]{0};
    for(int i=0;i<edges;i++){
        h_start_global[i]=edge_list[i].start;
        h_end_global[i]=edge_list[i].end;    
        h_cluster[i]=edge_list[i].cluster;
        h_start_global_2[i]=edge_list[i].start;
    }
    thrust::stable_sort_by_key(h_start_global,h_start_global+edges,h_end_global);
    thrust::stable_sort_by_key(h_start_global_2,h_start_global_2+edges,h_cluster);
    for(int i=0; i<edges;i++){
        edge_list[i].start=h_start_global[i];
        edge_list[i].end=h_end_global[i];
        edge_list[i].cluster=h_cluster[i];
    }
    delete [] h_start_global_2;
    delete [] h_cluster;
    Org_Vertex_Helper(edge_list,h_replica,deg,h_ctr,h_ptr,edges,nodes);
    h_ptr[BLOCKS]=h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1];
    cout<<"Ending Helper Function"<<endl;
    cpu_radixsort(edge_list,edges);
    Check_Out_csv_edge(edge_list, edges);
    check_out_replicas(REPLICA_PATH,h_replica,nodes);
    unsigned int* h_unique_merge = new unsigned int[2*edges]{0}; // used to merge these two arrays
    unsigned int* unq_ctr = new unsigned int[BLOCKS]{0};
    unsigned int* unq_ptr = new unsigned int[BLOCKS+1]{0};
    unsigned int** unq_fin = new unsigned int*[BLOCKS]{0};
    for(int i=0; i<BLOCKS;i++){
        cout<<h_ctr[i]<<endl;
    }
    Generate_Global_Src_Succ(h_start_global,h_end_global,src,succ,nodes,edges);
    delete[] h_start_global;
    delete[] h_end_global;
    for(int i=0; i<edges;i++){
        h_start[i]=edge_list[i].start;
        h_end[i]=edge_list[i].end;
    }
    cout<<"Starting the merge function"<<endl;
    for(int i = 0; i<BLOCKS;i++){
        sort(h_start+h_ptr[i],h_start+h_ptr[i]+h_ctr[i]);
        sort(h_end+h_ptr[i],h_end+h_ptr[i]+h_ctr[i]);
        // merge_sequential(h_start+h_ptr[i],h_end+h_ptr[i],h_ctr[i],h_ctr[i],h_unique_merge+2*h_ptr[i]);
        merge(h_start+h_ptr[i],h_start+h_ptr[i]+h_ctr[i],h_end+h_ptr[i],h_end+h_ptr[i]+h_ctr[i],h_unique_merge+2*h_ptr[i]);
        auto ip=unique(h_unique_merge+2*h_ptr[i],h_unique_merge+2*h_ptr[i]+2*h_ctr[i]);
        unq_ctr[i]=distance(h_unique_merge+2*h_ptr[i],ip);
        unq_fin[i]=new unsigned int[unq_ctr[i]]{0};
        copy(h_unique_merge+2*h_ptr[i],ip,unq_fin[i]);
    }
    cout<<"Ending the merge function"<<endl;
    unq_ptr[0]=0;
    for(int i = 1; i<=BLOCKS;i++){
        unq_ptr[i]=unq_ptr[i-1]+unq_ctr[i-1];
    }
    //No, we have new pointer to the unique array
    unsigned int* h_unq_fin = new unsigned int[unq_ptr[BLOCKS]];
    cout<<unq_ptr[BLOCKS]<<endl;
    for(int i = 0; i<BLOCKS;i++){
        for(int j = 0; j<unq_ctr[i];j++){
            h_unq_fin[unq_ptr[i]+j]=unq_fin[i][j];
        }
    }
    //Now, we have the unique array and the renumbering
    edge* edge_list_2;
    edge_list_2=(edge*)malloc(sizeof(edge)*edges);
    unsigned int* src_ptr = new unsigned int[BLOCKS+1]{0};
    unsigned int* src_ctr = new unsigned int[BLOCKS]{0};
    for(int i=0; i<BLOCKS;i++){
        src_ctr[i]=unq_ctr[i]+1;
    }
    src_ptr[0]=0;
    for(int i=1;i<=BLOCKS;i++){
        src_ptr[i]=src_ptr[i-1]+src_ctr[i-1];
    }
    unsigned int unq_ctr_max=0;
    unsigned int src_ctr_max=0;
    unsigned int h_ctr_max=0;
    for(int i = 0; i<BLOCKS;i++){
        if(unq_ctr[i]>unq_ctr_max){
            unq_ctr_max=unq_ctr[i];
        }
        if(h_ctr[i]>h_ctr_max){
            h_ctr_max=h_ctr[i];
        }
        if(src_ctr[i]>src_ctr_max){
            src_ctr_max=src_ctr[i];
        }
    }
    cout<<"Unq Ptr Values"<<endl;
    cout<<"Done making src, unq and succ"<<endl;
    unsigned int* h_local_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]]{0};
    unsigned int* h_temp_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]]{0};
    unsigned int* h_local_succ = new unsigned int[h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1]]{0};
    cout<<"Starting the generate renum edgelists"<<endl;
    Generate_Renum_Edgelists(edge_list, edge_list_2, h_unq_fin,h_ptr,h_ctr,unq_ctr,unq_ptr);
    Check_Out_Renum_Edge(edge_list_2,edges);
    cout<<"Done generating renum edgelists"<<endl;
    cout<<"Starting the generate local src and succ"<<endl;
    Gen_Local_Src_Succ(edge_list_2, h_local_src, h_temp_src, h_local_succ,src_ptr, h_unq_fin,unq_ctr,unq_ptr,h_ctr,h_ptr);
    cout<<"Local src values"<<endl;
    cout<<"Done generating local src and succ"<<endl;
    // Determine_Master(unq_ptr,h_replica,nodes);
    unsigned int* rank = new unsigned int[nodes];
    unsigned int* K = new unsigned int[nodes]{0};
    unsigned int* C = new unsigned int[nodes]{0};
    FrogWild(h_local_succ, h_local_src, h_unq_fin, C, K, src_ptr, unq_ptr, h_ptr,deg,src,succ,h_replica, nodes,edges,unq_ctr_max,1, rank,1);
    Export_C(C,rank,nodes);
    Export_K(K,nodes);
    Export_Local_Src(h_local_src,src_ptr,src_ctr);
    Export_Local_Succ(h_local_succ,h_ptr,h_ctr);
    Export_Unq(h_unq_fin,unq_ptr,unq_ctr);
    Export_Unq_Ctr_Ptr(unq_ptr,unq_ctr);
    Export_Src_Ctr_Ptr(src_ptr,src_ctr);
    Export_H_Ctr_Ptr(h_ptr,h_ctr);
    Export_Replica_Stats(h_replica,nodes);
    cout<<"Done exporting"<<endl;
    delete[] h_temp_src;
    free(h_replica);
    free(edge_list);
    free(edge_list_2);
    delete[] src;
    delete[] succ;
    delete[] deg;
    delete[] replica;
    delete[] h_ctr;
    delete[] h_ptr;
    delete[] h_unq;
    delete[] h_start;
    delete[] h_end;
    delete[] h_unique_merge;
    delete[] unq_fin;
    delete[] unq_ctr;
    delete[] unq_ptr;
    delete[] h_unq_fin;
    delete[] h_local_src;
    delete[] h_local_succ;
    delete[] src_ptr;
    delete[] src_ctr;
    delete[] rank;
    delete[] K;
    delete[] C;
    cout<<"Done"<<endl;
    return 0;
}