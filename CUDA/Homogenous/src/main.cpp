#include "../include/data.h"

// #define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
// #define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned long int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH,&nodes,&edges);
    cout<<edges<<endl;
    edge* edge_list;
    unsigned int* src_ptr = new unsigned int[nodes+1];
    unsigned int* succ = new unsigned int[edges];
    CSR_Graph(EDGE_PATH,nodes, edges, src_ptr,succ);
    edge_list=(edge*)malloc(sizeof(edge)*edges);
    cout<<"Starting the edge list function"<<endl;
    return_edge_list(EDGE_PATH,edge_list);
    cout<<"Ending edge list function"<<endl;
    // graph* G = create_graph(edge_list);
    cout<<"Starting Helper Function"<<endl;
    Org_Vertex_Helper(edge_list,edges);
    // cpu_radixsort(edge_list,edges);
    cout<<"Ending Helper Function"<<endl;
    Check_Out_csv_edge(edge_list, edges);

    // for (int i = 0; i < 512 ; i++){
    //     cout<<edge_list[i].cluster<<endl;
    // }
    free(edge_list);
    return 0;
}