#include "../include/data.h"
#include "../include/GPUErrors.h"


__host__ void Import_Local_Src(unsigned int* local_src, unsigned int* src_ptr, unsigned int* src_ctr){
    ifstream myfile;
    myfile.open(LOCAL_SRC_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        src_ctr[stoi(word)]++;
                    }
                    else{
                        local_src[count-1] = stoi(word);
                    }
                }
            }
        }
        src_ptr[0] = 0;
        for(int i=1; i<BLOCKS;i++){
            src_ptr[i] = src_ptr[i-1] + src_ctr[i-1];
        }
    }
}


__host__ void Import_Local_Succ(unsigned int* local_succ, unsigned int* succ_ptr, unsigned int* succ_ctr){
    ifstream myfile;
    myfile.open(LOCAL_SUCC_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        succ_ctr[stoi(word)]++;
                    }
                    else{
                        local_succ[count-1] = stoi(word);
                    }
                }
            }
        }
        succ_ptr[0] = 0;
        for(int i=1; i<BLOCKS;i++){
            succ_ptr[i] = succ_ptr[i-1] + succ_ctr[i-1];
        }
    }
}

__host__ void Import_Unique(unsigned int* unq){
    ifstream myfile;
    myfile.open(UNQ_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else{
                        unq[count-1] = stoi(word);
                    }
                }
            }
            column = 0;
            count++;
        }
    }
}



__host__ void Import_Src_Ctr_Ptr(unsigned int* src_ctr, unsigned int* src_ptr){
    ifstream myfile;
    myfile.open(SRC_CTR_PTR_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        src_ctr[count-1] = stoi(word);
                    }
                    else{
                        src_ptr[count-1] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }

}


__host__ void Import_Unq_Ptr_Ctr(unsigned int* unq_ptr, unsigned int* unq_ctr){
    ifstream myfile;
    myfile.open(UNQ_CTR_PTR_PATH);
    string line,word;
    int count = 0;
    int column = 0;
    if(!myfile.is_open()){
        cout << "Error opening file" << endl;
        exit(1);
    }
    else{
        while(getline(myfile,line)){
            stringstream s(line);
            while(getline(s,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    if(column==0){
                        column++;
                    }
                    else if(column==1){
                        unq_ctr[count-1] = stoi(word);
                    }
                    else{
                        unq_ptr[count-1] = stoi(word);
                    }
                }
            }
            count++;
            column = 0;
        }
    }

}


__global__ void FrogWild(unsigned int* local_succ, unsigned int* local_src, unsigned int* unq, unsigned int* c, unsigned int* k, float ps, float pt){

}