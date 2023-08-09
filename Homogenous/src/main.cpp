#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned int** edges=new unsigned int*[EDGES];
    for(int i=0; i<EDGES; i++){
        edges[i]=new unsigned int[2];
    }
    return_list(EDGE_PATH,edges);

    struct edge e[EDGES]=edges;

    for(int i=0; i<EDGES;i++){
        delete[] edges[i];
    }


    delete[] edges;
    return 0;
}