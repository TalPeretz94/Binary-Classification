#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "point.h"
#include <stdio.h>
#include <omp.h>
#include <mpi.h>
#include "algorithm.h"
#define FILE_PATH "C:\\Users\\talpe\\Desktop\\finalProject\\CudaMPIOpenMP_onVDI\\dataSetOne.txt" //Should change file Path
#define FILE_RESULT "C:\\Users\\talpe\\Desktop\\finalProject\\CudaMPIOpenMP_onVDI\\results.txt" //Should change file Path








void writeToFile(int K, float alpha, float* w, float q, float qc) { //write results to the file
	FILE *f = fopen(FILE_RESULT, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	/* print integers and floats */
	if (q < qc) {
		fprintf(f, "Alpha minimum = %f  ", alpha);
	}
	else
		fprintf(f, "Alpha is not found  ");

	fprintf(f, "The q: %f\n", q);
	for (int i = 0; i < K + 1; i++) {
		fprintf(f, "W%d = %f\n", i, w[i]);
	}
	
	fclose(f);
}


Point* readFromFile(int *N, int *K, float *alpha, float *alphaMax, int *limit, float *qc) {// read all the parameters and points

	FILE *file;
	printf_s("open file from %s \n", FILE_PATH); _flushall();
	file = fopen(FILE_PATH, "r");
	if (file == NULL)
	{
		printf_s("Failed to open file \n"); _flushall();
		_flushall();
	}
	fscanf(file, "%d %d %f %f %d %f", N, K, alpha, alphaMax, limit, qc);

	Point* myArray = (Point*)malloc(sizeof(Point) * (*N));
	for (int i = 0; i < *N; i++)
	{
		myArray[i].value = (float*) malloc((*K + 1) * sizeof(float));
		for (int j = 0; j < *K; j++)
			fscanf(file, "%f", &myArray[i].value[j]);
		myArray[i].value[*K] = 1;// adding bias
		fscanf(file, "%d", &myArray[i].pointClass);
	}
	fclose(file);
	return myArray;
}


int main(int argc, char *argv[])
{
	float alphaZero, alphaMax, qc;
	int N, K, limit, nMiss;
	printf("reading data from file\n");
	Point* arr = readFromFile(&N, &K, &alphaZero, &alphaMax, &limit, &qc);// create array with the points from file
	float q, alphaMin;
	printf("execute algorithm\n");
	float* w = algo(&q, &alphaMin, K, N, arr, alphaZero, alphaMax, limit, qc);

	
	printf("write to file\n");
	writeToFile(K, alphaMin, w, q, qc);

	system("pause");
}
