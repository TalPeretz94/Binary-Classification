#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "point.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

float* algo(float *q, float* alphaMin, int K, int N, Point* arr, float alphaZero, float alphaMax, int limit, float qc);