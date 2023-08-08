#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned int** edges=new unsigned int*[EDGES];
    for(int i=0; i<EDGES; i++){
        edges[i]=new unsigned int[2];
    }
    unsigned int** cluster_vert=new unsigned int*[NODES];
    for(int i=0; i<NODES;i++){
        cluster_vert[i]=new unsigned int[2];
    }
    return_list(EDGE_PATH,edges);
    return_list(CLUSTER_PATH,cluster_vert);
    unsigned int* cluster_assign= unsigned new int[NODES];
    unsigned int* vert_assign=new unsigned int[NODES];
    split_list(cluster_vert,vert_assign,cluster_assign,NODES);
    Org_Vertex_Helper(cluster_assign,vert_assign,NODES);

    for(int i = 0; i<NODES ; i++){
        cout<<cluster_assign[i]<<'\t'<<vert_assign[i]<<endl;
    }
    for(int i=0; i<NODES;i++){
        delete[] cluster_vert[i];
    }

    for(int i=0; i<EDGES;i++){
        delete[] edges[i];
    }


    delete[] cluster_vert;
    delete[] vert_assign;
    delete[] cluster_assign;
    delete[] edges;
    return 0;
}