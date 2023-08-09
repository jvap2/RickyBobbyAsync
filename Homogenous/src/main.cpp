#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    edge* edge_list;
    edge_list=malloc(sizeof(edge)*EDGES);
    return_list(EDGE_PATH,edge_list);


    for(int i=0; i<EDGES;i++){
        delete[] edges[i];
    }


    delete[] edges;
    return 0;
}