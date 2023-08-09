#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    edge* edge_list;
    edge_list=(edge*)malloc(sizeof(edge)*EDGES);
    return_edge_list(EDGE_PATH,edge_list);
    free(edge_list);
    return 0;
}