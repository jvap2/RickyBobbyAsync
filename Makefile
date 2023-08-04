PLATFORM = $(shell uname -m)
CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
NVCC := nvcc

NVCC_FLAGS := -g -G -Xcompiler -Wall -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89

INC := include
OBJ := obj
SRC := src
BIN := bin


$(BIN)/Frog: $(OBJ)/frog.o 
	$(NVCC) $(LINK_FLAGS) $^ -o $@
	
$(OBJ)/frog.o: $(SRC)/main_sparse.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/SVM_sparse.o: $(SRC)/SVM_sparse.cu $(INC)/SVM_sparse.cuh 
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ)/sparse_data.o: $(SRC)/sparse_data.cu $(INC)/sparse_data.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@