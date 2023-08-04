#include "data.h"



__host__ void return_list(string path, int** arr){
    fstream data;
    data.open(path);
    string line,word;
    int count=0;
    if(data.is_open()){
        //Check if data is open
        while(getline(data,line)){
            //Keep extracting data until a delimiter is found
            stringstream stream_data(line); //Stream Class to operate on strings
            while(getline(stream_data,word,',')){
                if(count==0){
                    continue;
                }
                else{
                    *(arr[count-1])=stoi(word);
                    arr[count-1]++;
                }
                //Extract data until ',' is found
            }
            count++;
        }
    }
    data.close();
}

