#include "../include/data.h"

// #define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
// #define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    cout<<nodes<<endl;
    cout<<edges<<endl;
    edge* edge_list;
    unsigned int* src_ptr = new unsigned int[nodes+1];
    unsigned int* succ = new unsigned int[edges];
    unsigned int* deg = new unsigned int[nodes];
    // CSR_Graph(GRAPH_DATA_PATH ,nodes, edges, src_ptr,succ,deg);
    Check_Out_ptr(src_ptr, nodes+1);
    edge_list=(edge*)malloc(sizeof(edge)*edges);
    unsigned int* replica = new unsigned int[edges];
    cout<<"Starting the edge list function"<<endl;
    return_edge_list(EDGE_PATH,edge_list);
    Capture_Node_Degree(edge_list,deg,edges);
    unsigned int *h_ctr, *h_ptr;
    h_ctr = new unsigned int[BLOCKS];
    h_ptr = new unsigned int[BLOCKS];
    unsigned int *h_unq;
    h_unq = new unsigned int[edges];
    cout<<"Ending edge list function"<<endl;
    cout<<"Starting Helper Function"<<endl;
    Org_Vertex_Helper(edge_list,replica,deg,h_ctr,h_ptr,edges,nodes);
    cout<<"Ending Helper Function"<<endl;
    for(int i = 0; i<BLOCKS;i++){
        cout<<h_ctr[i]<<"\t"<<h_ctr[i]+h_ptr[i]<<endl;
    }
    cpu_radixsort(edge_list,edges);
    Check_Out_csv_edge(edge_list, edges);
    Check_Repeats(edge_list,edges);
    check_out_replicas(REPLICA_PATH,replica,nodes);
    Check_Out_Ptr_Ctr(h_ctr,h_ptr,BLOCKS);
    unsigned int* h_start = new unsigned int[edges];
    unsigned int* h_end = new unsigned int[edges];
    unsigned int* h_unique_merge = new unsigned int[2*edges];
    unsigned int* unq_ctr = new unsigned int[BLOCKS];
    unsigned int* unq_ptr = new unsigned int[BLOCKS];
    unsigned int** unq_fin = new unsigned int*[BLOCKS];

    cout<<"Copying the edge list"<<endl;
    for(int i=0;i<edges;i++){
        h_start[i]=edge_list[i].start;
        h_end[i]=edge_list[i].end;
    }
    cout<<"Starting the merge function"<<endl;
    for(int i = 0; i<BLOCKS;i++){
        sort(h_start+h_ptr[i],h_start+h_ptr[i]+h_ctr[i]);
        sort(h_end+h_ptr[i],h_end+h_ptr[i]+h_ctr[i]);
        merge_sequential(h_start+h_ptr[i],h_end+h_ptr[i],h_ctr[i],h_ctr[i],h_unique_merge+2*h_ptr[i]);
        auto ip=unique(h_unique_merge+2*h_ptr[i],h_unique_merge+2*h_ptr[i]+2*h_ctr[i]);
        unq_ctr[i]=distance(h_unique_merge+2*h_ptr[i],ip);
        unq_fin[i]=new unsigned int[unq_ctr[i]];
        copy(h_unique_merge+2*h_ptr[i],ip,unq_fin[i]);
    }
    cout<<"Ending the merge function"<<endl;
    unq_ptr[0]=0;
    for(int i = 1; i<BLOCKS;i++){
        unq_ptr[i]=unq_ptr[i-1]+unq_ctr[i-1];
    }
    
    //Now, there needs to be a mapping
    free(edge_list);
    delete[] src_ptr;
    delete[] succ;
    delete[] deg;
    delete[] replica;
    delete[] h_ctr;
    delete[] h_ptr;
    delete[] h_unq;
    delete[] h_start;
    delete[] h_end;
    delete[] h_unique_merge;
    return 0;
}