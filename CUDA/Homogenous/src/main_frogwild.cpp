#include "../include/data.h"

int main(int argc, char** argv){
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    unsigned int *src_ptr, *src_ctr;
    unsigned int *unq_ptr, *unq_ctr;
    unsigned int *h_ptr, *h_ctr;

    src_ptr = new unsigned int[BLOCKS];
    src_ctr = new unsigned int[BLOCKS];
    unq_ptr = new unsigned int[BLOCKS+1];
    unq_ctr = new unsigned int[BLOCKS+1];
    h_ptr = new unsigned int[BLOCKS];
    h_ctr = new unsigned int[BLOCKS];

    Import_Unq_Ptr_Ctr(unq_ptr,unq_ctr);
    Import_Src_Ctr_Ptr(src_ctr,src_ptr);
    Import_H_Ctr_Ptr(h_ptr,h_ctr);
    unsigned int* h_unq = new unsigned int[unq_ptr[BLOCKS-1]+unq_ctr[BLOCKS-1]];
    unsigned int* h_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]];
    unsigned int* h_succ = new unsigned int[h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1]];
    

    return 0;
}