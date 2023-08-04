#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
using namespace std;
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "GPUErrors.h"
#define EDGES 5066842
#define NODES 855802
#define EDGE_PATH "../Data/webGoogle.csv"
#define CLUSTER_PATH "../Data/Cluster_Assignment.csv.csv"

void return_list(string path, int** arr);