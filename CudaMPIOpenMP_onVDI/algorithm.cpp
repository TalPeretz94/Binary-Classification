#include "algorithm.h"


int CalcNmissPoints(const int size, Point* point, float *w, const int k);
Point* initCudaMalloc(const int N, Point* point, const int k);
bool SameSign(float x, float y);

bool SameSign(float x, float y)
{
	return (x >= 0) ^ (y < 0);
}

int nMissCalcOMP(int N, int K, Point *arr, float* w) {// optinal
	int nMiss = 0;
#pragma omp parallel for schedule(dynamic) reduction(+: nMiss) // Calculate the amount of points that were not properly classified with OMP 
	for (int i = 0; i < N; i++) {
		float result = 0;
		for (int j = 0; j < K + 1; j++) {
			result = result + (arr[i].value[j] * w[j]);//cal F(Xi)

		}
		if (!(SameSign(arr[i].pointClass, result))) { //check if F(Xi) sign not equals to Point sign
			nMiss++;

		}
	}
	return nMiss;

}




float* algo(float *q, float* alphaMin, int K , int N, Point* arr, float alphaZero, float alphaMax,int limit, float qc) {

	float *w = (float *)malloc((K + 1) * sizeof(float));//creating w array

	Point* cudaPoints = initCudaMalloc(N, arr, K);
	float alpha;
	int nMiss;

	for (alpha = alphaZero; alpha < alphaMax; alpha += alphaZero) // run alpha Iterations according to alphazero and alphamax from file
	{
#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < K + 1; i++) { //initialize weight array with zeros
			w[i] = 0;
		}
		//printf("the alpha - %f\n", alpha);

		for (int counter = 0; counter < limit; counter++)
		{

			bool isAllGood = true;// initialize point classifid flag

			for (int i = 0; i < N; i++) {
				float result = 0;
#pragma omp parallel for schedule(dynamic) reduction(+: result)
				for (int j = 0; j < K + 1; j++) {// calculate f(Xi) for each Point according the formola
					result = result + (arr[i].value[j] * w[j]);
				}
				if (!(SameSign(arr[i].pointClass, result))) { //checking if the sign F(Xi) equals to the point sing (-1 or 1)
					int mult = result >= 0 ? 1 : -1;
					isAllGood = false;
					
#pragma omp parallel for schedule(dynamic)
					for (int z = 0; z < K + 1; z++) //change the weight function because the point not properly classified
						w[z] = w[z] + (alpha*mult)*arr[i].value[z];
					break; //break the loop for running all the point from the beginning with the update weight(W)
				}


			}

			if (isAllGood)// in case all the points classified correctly
				break;

		}
		nMiss = 0;
		nMiss = CalcNmissPoints(N, cudaPoints, w, K); // Calculate the amount of points that were not properly classified with cuda

													  //nMiss = nMissCalcOMP(N, K, arr,w); //Calculate the amount of points that were not properly classified with OMP 


		*q = (float)nMiss / (float)N;
		
		if (*q < qc) {
			break;

		}
	}
	*alphaMin = alpha;
	return w;
}