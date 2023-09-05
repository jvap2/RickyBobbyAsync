#include "../include/data.h"

int main(int argc, char** argv){
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    unsigned int *src_ptr, *src_ctr;
    unsigned int *unq_ptr, *unq_ctr;

    src_ptr = new unsigned int[BLOCKS];
    src_ctr = new unsigned int[BLOCKS];
    unq_ptr = new unsigned int[BLOCKS+1];
    unq_ctr = new unsigned int[BLOCKS];
    

    return 0;
}