#include "../include/data.h"

int main(int argc, char** argv){
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    unsigned int *src_ptr, *src_ctr;
    unsigned int *unq_ptr, *unq_ctr;
    unsigned int *h_ptr, *h_ctr;
    unsigned int *K, *C;

    unsigned int* global_src = new unsigned int[nodes+1];
    unsigned int* global_succ = new unsigned int[edges];
    Import_Global_Src(global_src);
    Import_Global_Succ(global_succ);

    src_ptr = new unsigned int[BLOCKS+1];
    src_ctr = new unsigned int[BLOCKS];
    unq_ptr = new unsigned int[BLOCKS+1];
    unq_ctr = new unsigned int[BLOCKS+1];
    h_ptr = new unsigned int[BLOCKS+1];
    h_ctr = new unsigned int[BLOCKS];
    K = new unsigned int[nodes]{0};
    C = new unsigned int[nodes]{0};


    Import_Unq_Ptr_Ctr(unq_ptr,unq_ctr);
    Import_Src_Ctr_Ptr(src_ctr,src_ptr);
    Import_H_Ctr_Ptr(h_ctr,h_ptr);
    unsigned int* h_unq = new unsigned int[unq_ptr[BLOCKS-1]+unq_ctr[BLOCKS-1]];
    unsigned int* h_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]];
    unsigned int* h_succ = new unsigned int[h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1]];
    Import_Unique(h_unq);
    Import_Local_Src(h_src);
    Import_Local_Succ(h_succ);
    src_ptr[BLOCKS]=src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1];
    unq_ptr[BLOCKS]=unq_ptr[BLOCKS-1]+unq_ctr[BLOCKS-1];
    h_ptr[BLOCKS]=h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1];
    unsigned int max_unq_ctr=0;
    for(int i=0;i<BLOCKS;i++){
        if(unq_ctr[i]>max_unq_ctr){
            max_unq_ctr=unq_ctr[i];
        }
    }
    // for(int i=0; i<=BLOCKS;i++){
    //     // cout<<src_ptr[i]<<endl;
    //     cout<<unq_ptr[i]<<endl;
    //     // cout<<h_ptr[i]<<endl;
    // }

    replica_tracker* h_replica = (replica_tracker*)malloc(sizeof(replica_tracker)*nodes);
    Import_Replica_Stats(h_replica,nodes);
    unsigned int* deg = new unsigned int[nodes];
    Import_Degree(deg,nodes);
    FrogWild(h_succ, h_src, h_unq, C, K, src_ptr, unq_ptr, h_ptr,deg,global_src,global_succ,h_replica, nodes,edges,max_unq_ctr,0);
    Export_C(C,nodes);
    Export_K(K,nodes);
    return 0;
}