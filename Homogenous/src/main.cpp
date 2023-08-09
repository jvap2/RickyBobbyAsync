#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    edge* edge_list;
    edge_list=(edge*)malloc(sizeof(edge)*EDGES);
    return_edge_list(EDGE_PATH,edge_list);
    graph* G = create_graph(edge_list);
    cout<<"Starting Helper Function"<<endl;
    Org_Vertex_Helper(edge_list,EDGES);
    cout<<"Ending Helper Function"<<endl;
    free(edge_list);
    return 0;
}