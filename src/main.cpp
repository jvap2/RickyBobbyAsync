#include "../include/data.h"


int main()
{  
    int** edges=new int*[EDGES];
    for(int i=0; i<EDGES; i++){
        edges[i]=new int[2];
    }
    int** cluster=new int*[NODES];
    for(int i=0; i<NODES;i++){
        cluster[i]=new int[2];
    }
    return_list(EDGE_PATH,edges);
    return_list(CLUSTER_PATH,cluster);

    return 0;
}