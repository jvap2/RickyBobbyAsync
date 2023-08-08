#include "../include/data.h"

#define EDGE_PATH "../Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Data/google/Cluster_Assignment.csv"

int main()
{  
    int** edges=new int*[EDGES];
    for(int i=0; i<EDGES; i++){
        edges[i]=new int[2];
    }
    int** cluster_vert=new int*[NODES];
    for(int i=0; i<NODES;i++){
        cluster_vert[i]=new int[2];
    }
    return_list(EDGE_PATH,edges);
    return_list(CLUSTER_PATH,cluster_vert);
    cout<<"Cluster Vert"<<endl;
    for(int i = 0; i<32 ; i++){
        cout<<cluster_vert[i][0]<<'\t'<<cluster_vert[i][1]<<endl;
    }
    int* cluster_assign= new int[NODES];
    int* vert_assign=new int[NODES];
    split_list(cluster_vert,vert_assign,cluster_assign,NODES);
    cout<<"Split Output"<<endl;
    for(int i = 0; i<32 ; i++){
        cout<<cluster_assign[i]<<'\t'<<vert_assign[i]<<endl;
    }
    Org_Vertex_Helper(cluster_assign,vert_assign,NODES);

    for(int i = 0; i<32 ; i++){
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