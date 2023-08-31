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
    Org_Vertex_Helper(edge_list,replica,deg,edges,nodes);
    // cpu_radixsort(edge_list,edges);
    cout<<"Ending Helper Function"<<endl;
    Check_Out_csv_edge(edge_list, edges);
    check_out_replicas(REPLICA_PATH,replica,nodes);
    free(edge_list);
    delete[] src_ptr;
    delete[] succ;
    delete[] deg;
    delete[] replica;
    delete[] h_ctr;
    delete[] h_ptr;
    delete[] h_unq;
    return 0;
}