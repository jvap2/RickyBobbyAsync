#include "../include/data.h"

#define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
#define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    edge* edge_list;
    edge_list=(edge*)malloc(sizeof(edge)*EDGES);
    return_edge_list(EDGE_PATH,edge_list);
    for(int i = 0; i<32 ; i++){
        cout<<edge_list[i].start<<endl;
        cout<<edge_list[i].end<<endl;
    }
    free(edge_list);
    return 0;
}