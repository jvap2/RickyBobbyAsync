#include "../include/data.h"

// #define EDGE_PATH "../Homogenous/Data/google/webGoogle.csv"
// #define CLUSTER_PATH "../Homogenous/Data/google/Cluster_Assignment.csv"

int main()
{  
    unsigned int nodes, edges;
    get_graph_info(GRAPH_DATA_PATH ,&nodes,&edges);
    cout<<nodes<<endl;
    cout<<edges<<endl;
    edge* edge_list;
    unsigned int* deg = new unsigned int[nodes];
    edge_list=(edge*)malloc(sizeof(edge)*edges);
    unsigned int* replica = new unsigned int[edges];
    cout<<"Starting the edge list function"<<endl;
    return_edge_list(EDGE_PATH,edge_list);
    Capture_Node_Degree(edge_list,deg,edges);
    unsigned int *h_ctr, *h_ptr;
    h_ctr = new unsigned int[BLOCKS];
    h_ptr = new unsigned int[BLOCKS];
    unsigned int *h_unq;
    h_unq = new unsigned int[edges];
    cout<<"Ending edge list function"<<endl;
    cout<<"Starting Helper Function"<<endl;
    Org_Vertex_Helper(edge_list,replica,deg,h_ctr,h_ptr,edges,nodes);
    cout<<"Ending Helper Function"<<endl;
    for(int i = 0; i<BLOCKS;i++){
        cout<<h_ctr[i]<<"\t"<<h_ctr[i]+h_ptr[i]<<endl;
    }
    cpu_radixsort(edge_list,edges);
    Check_Out_csv_edge(edge_list, edges);
    Check_Repeats(edge_list,edges);
    check_out_replicas(REPLICA_PATH,replica,nodes);
    Check_Out_Ptr_Ctr(h_ctr,h_ptr,BLOCKS);
    unsigned int* h_start = new unsigned int[edges]{0};
    unsigned int* h_end = new unsigned int[edges]{0};
    unsigned int* h_unique_merge = new unsigned int[2*edges]{0};
    unsigned int* unq_ctr = new unsigned int[BLOCKS]{0};
    unsigned int* unq_ptr = new unsigned int[BLOCKS+1]{0};
    unsigned int** unq_fin = new unsigned int*[BLOCKS]{0};

    cout<<"Copying the edge list"<<endl;
    for(int i=0;i<edges;i++){
        h_start[i]=edge_list[i].start;
        h_end[i]=edge_list[i].end;
    }
    cout<<"Starting the merge function"<<endl;
    for(int i = 0; i<BLOCKS;i++){
        sort(h_start+h_ptr[i],h_start+h_ptr[i]+h_ctr[i]);
        sort(h_end+h_ptr[i],h_end+h_ptr[i]+h_ctr[i]);
        merge_sequential(h_start+h_ptr[i],h_end+h_ptr[i],h_ctr[i],h_ctr[i],h_unique_merge+2*h_ptr[i]);
        auto ip=unique(h_unique_merge+2*h_ptr[i],h_unique_merge+2*h_ptr[i]+2*h_ctr[i]);
        unq_ctr[i]=distance(h_unique_merge+2*h_ptr[i],ip);
        unq_fin[i]=new unsigned int[unq_ctr[i]];
        copy(h_unique_merge+2*h_ptr[i],ip,unq_fin[i]);
    }
    cout<<"Ending the merge function"<<endl;
    unq_ptr[0]=0;
    for(int i = 1; i<=BLOCKS;i++){
        unq_ptr[i]=unq_ptr[i-1]+unq_ctr[i-1];
    }
    //No, we have new pointer to the unique array
    unsigned int* h_unq_fin = new unsigned int[unq_ptr[BLOCKS]];
    for(int i = 0; i<BLOCKS;i++){
        for(int j = 0; j<unq_ctr[i];j++){
            h_unq_fin[unq_ptr[i]+j]=unq_fin[i][j];
        }
    }
    //Now, we have the unique array and the renumbering
    edge* edge_list_2;
    edge_list_2=(edge*)malloc(sizeof(edge)*edges);
    unsigned int* src_ptr = new unsigned int[BLOCKS]{0};
    unsigned int* src_ctr = new unsigned int[BLOCKS]{0};
    for(int i=0; i<BLOCKS;i++){
        src_ctr[i]=unq_ctr[i]+1;
    }
    src_ptr[0]=0;
    for(int i=1;i<BLOCKS;i++){
        src_ptr[i]=src_ptr[i-1]+src_ctr[i-1];
    }
    unsigned int* h_local_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]]{0};
    unsigned int* h_temp_src = new unsigned int[src_ptr[BLOCKS-1]+src_ctr[BLOCKS-1]]{0};
    unsigned int* h_local_succ = new unsigned int[h_ptr[BLOCKS-1]+h_ctr[BLOCKS-1]]{0};
    Generate_Renum_Edgelists(edge_list, edge_list_2, h_unq_fin,h_ptr,h_ctr,unq_ctr,unq_ptr);
    Gen_Local_Src(edge_list_2, h_local_src, h_temp_src, h_unq_fin,src_ctr,src_ptr,h_ctr,h_ptr);
    Generate_Local_Succ(edge_list_2, h_local_src, h_local_succ,src_ctr,src_ptr,h_ptr);
    Export_Local_Src(h_local_src,src_ptr,src_ctr);
    Export_Local_Succ(h_local_succ,h_ptr,h_ctr);
    delete[] h_temp_src;
    free(edge_list);
    free(edge_list_2);
    delete[] deg;
    delete[] replica;
    delete[] h_ctr;
    delete[] h_ptr;
    delete[] h_unq;
    delete[] h_start;
    delete[] h_end;
    delete[] h_unique_merge;
    delete[] unq_fin;
    delete[] unq_ctr;
    delete[] unq_ptr;
    delete[] h_unq_fin;
    delete[] h_local_src;
    delete[] h_local_succ;
    delete[] src_ptr;
    delete[] src_ctr;
    return 0;
}