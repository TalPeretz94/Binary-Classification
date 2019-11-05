
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "point.h"
#include <stdio.h>

#define THREAD_AMOUNT 1024


Point* initCudaMalloc(const int N, Point* point, const int k);
int CalcNmissPoints(const int N, Point* point, float *w, const int k);

__device__ int getThreadId()
{
	return blockIdx.x *blockDim.x + threadIdx.x; //returning theard num (from internet)
}

__global__ void backprobagation(float *w, const float *point, const unsigned int size, const float alpha, const int sign)// update W in case of point not properly classified
{
	int i = getThreadId();
	if (i < size)
		w[i] = w[i] + alpha*sign*point[i];
}

__global__ void calcMiss(Point* point, float *w, const int k, const int N, int* nMiss)//each thread calc f(Xi) check if the signs are equals in case the sign not equals
																					  //the point not properly classified so we increase the amount of Nmiss.	
{
	int i = getThreadId();
	

	if (i < N) {
		float result = 0;
		for (int j = 0; j < k + 1; j++) {
			result = result + (point[i].value[j] * w[j]);
			
		}
		if (result*point[i].pointClass < 0) {
			//point[i].cuda = 1;
			atomicAdd(nMiss, 1);
			

		}
//		else
			//point[i].cuda = result; //

	}
}




Point* initCudaMalloc(const int N, Point* point, const int k) {
	Point *cuda_point;
	cudaMalloc(&cuda_point, N * sizeof(Point));
	cudaMemcpy(cuda_point, point, N * sizeof(Point), cudaMemcpyHostToDevice);
	int i = 0;
	for (i = 0; i < N; i++) {
		float* arr;
		cudaMalloc(&arr, (k + 1) * sizeof(float));
		cudaMemcpy(&(cuda_point[i].value), &arr, sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(arr, point[i].value, (k + 1) * sizeof(float), cudaMemcpyHostToDevice);
	}
	return cuda_point;
}

int CalcNmissPoints(const int N, Point* point, float *w, const int k)
{
	float *cuda_w;
	
	int nMiss = 0;
	int *nMissCuda;
	cudaMalloc(&cuda_w, (k+1) * sizeof(float));
	
	cudaMalloc(&nMissCuda, sizeof(int));

	cudaMemcpy(cuda_w, w, (k + 1) * sizeof(float), cudaMemcpyHostToDevice);
	
	

	cudaMemcpy(nMissCuda, &nMiss, sizeof(int), cudaMemcpyHostToDevice);

	calcMiss << <N / THREAD_AMOUNT + 1, THREAD_AMOUNT >> > (point, cuda_w, k, N, nMissCuda);

	cudaMemcpy(&nMiss, nMissCuda, sizeof(int), cudaMemcpyDeviceToHost);
	/*cudaMemcpy(point, cuda_point, N * sizeof(Point), cudaMemcpyDeviceToHost);
	for (int i = 0; i < k; i++)
		cudaMemcpy(point[i].po, cuda_point[i].po, N * sizeof(Point), cudaMemcpyDeviceToHost);*/

	cudaFree(cuda_w);
	
	cudaFree(nMissCuda);

	return nMiss;
}
