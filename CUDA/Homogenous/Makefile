PLATFORM = $(shell uname -m)
CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
NVCC := nvcc

NVCC_FLAGS := -g -G -Xcompiler -Wall -rdc=true -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89  
INC_FLAGS := -lcudadevrt -lcurand
# C_FLAGS := -std=c++11 

INC := include
OBJ := obj
SRC := src
BIN := bin

frog: $(BIN)/Frog

$(BIN)/Frog: $(OBJ)/frog.o $(OBJ)/frog_func.o $(OBJ)/cuda_error.o
	$(NVCC) $(LINK_FLAGS) $^ -o $@ -lcudadevrt
	
$(OBJ)/frog.o: $(SRC)/main.cpp $(INC)/data.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 

$(OBJ)/frog_func.o: $(SRC)/parse_data.cu $(INC)/data.h 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(INC_FLAGS)

$(OBJ)/cuda_error.o: $(INC)/GPUErrors.cu $(INC)/GPUErrors.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ 