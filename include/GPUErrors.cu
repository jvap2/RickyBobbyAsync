#include "GPUErrors.h"

bool HandleCUDAError(cudaError_t t)
{
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(cudaGetLastError());//This will get the string of the error for blocking error
		return false;
	}
	return true;
}
//We can have runtime errors on the GPU, which is what the function below is used for
bool GetCUDARunTimeError()
{
	cudaError_t t = cudaGetLastError();
	if (t != cudaSuccess)
	{
		cout << cudaGetErrorString(t) << endl;
		return false;
	}
	return true;
}

