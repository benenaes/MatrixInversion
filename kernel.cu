
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Utilities.cuh"

#include <thrust/device_vector.h>

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <utility>


#include <curand.h>
#include <cublas_v2.h>
#include <cusolverRf.h>
#include <cusolverDn.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void GpuGenerateRandomVector(
	float * devVector,
	const unsigned int numElements)
{
	// Create a pseudo-random number generator
	curandGenerator_t prng;

	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) std::clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, devVector, numElements);
}

void GpuGenerateRandomVector(
	thrust::device_vector<float> devVector)
{
	GpuGenerateRandomVector(thrust::raw_pointer_cast(devVector.data()), devVector.size());
}

__global__ 
void GpuInitIdentity(
	float *devMatrix, 
	const unsigned int numRows, 
	const unsigned int numCols) 
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (y < numRows && x < numCols) 
	{
		if (x == y)
		{
			devMatrix[IDX2C(x, y, y)] = 1.0;
		}
		else
		{
			devMatrix[IDX2C(x, y, y)] = 0.0;
		}
	}
}

void GpuBlasMatrixMultiply(
	cublasHandle_t &handle, 
	const float *A, 
	const float *B, 
	float *C, 
	const int m, 
	const int k, 
	const int n) 
{
	const int lda = m;
	const int ldb = k;
	const int ldc = m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

std::pair<float*, float**>
GpuCreatePositiveInvertibleSquareMatrices(
	cublasHandle_t &handle,
	const unsigned int dimension,
	const unsigned int numMatrices)
{
	//thrust::device_vector<float> devRandomMatrices(dimension * dimension * numMatrices);
	float* devRandomMatrices;
	gpuErrchk(cudaMalloc((void**)&devRandomMatrices, dimension * dimension * numMatrices * sizeof(float)));

	float* devInvertibleMatrices;
	gpuErrchk(cudaMalloc((void**)&devInvertibleMatrices, dimension * dimension * numMatrices * sizeof(float)));

	//float* rawRandomMatrix = thrust::raw_pointer_cast(devRandomMatrix.data());
	//float* rawOutputMatrix = thrust::raw_pointer_cast(devOutputMatrix.data());

	//GpuGenerateRandomVector(thrust::raw_pointer_cast(devRandomMatrices.data()), dimension * dimension * numMatrices);
	GpuGenerateRandomVector(devRandomMatrices, dimension * dimension * numMatrices);

	float** devRandomMatrixPointers;
	gpuErrchk(cudaMalloc((void**)&devRandomMatrixPointers, numMatrices * sizeof(float *)));

	float** devInvertibleMatrixPointers;
	gpuErrchk(cudaMalloc((void**)&devInvertibleMatrixPointers, numMatrices * sizeof(float *)));

	std::vector<float*> randomMatrixPointers(numMatrices);
	std::vector<float*> invertibleMatrixPointers(numMatrices);

	for (unsigned int index = 0; index < numMatrices; ++index)
	{
		//randomMatrixPointers[index] = thrust::raw_pointer_cast(devRandomMatrices.data()) + (index * dimension * dimension);
		randomMatrixPointers[index] = devRandomMatrices + (index * dimension * dimension);
		invertibleMatrixPointers[index] = devInvertibleMatrices + (index * dimension * dimension);
	}

	gpuErrchk(cudaMemcpy(devRandomMatrixPointers, randomMatrixPointers.data(), numMatrices * sizeof(float *), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(devInvertibleMatrixPointers, invertibleMatrixPointers.data(), numMatrices * sizeof(float *), cudaMemcpyHostToDevice));

	const float alpha = 1;
	const float beta = 0;

	// transpose(random matrix) * random matrix leads into an invertible, positive definite matrix
	cublasSafeCall(
		cublasSgemmBatched(
			handle, 
			CUBLAS_OP_T, 
			CUBLAS_OP_N, 
			dimension, 
			dimension, 
			dimension, 
			&alpha, 
			(const float**)devRandomMatrixPointers,
			dimension, 
			(const float**)devRandomMatrixPointers,
			dimension, 
			&beta, 
			devInvertibleMatrixPointers,
			dimension,
			numMatrices));

	gpuErrchk(cudaFree(devRandomMatrices));
	gpuErrchk(cudaFree(devRandomMatrixPointers));

	return std::make_pair(devInvertibleMatrices, devInvertibleMatrixPointers);
}

int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
	std::cout << "Max global memory: " << prop.totalGlobalMem << "\n";

	// Rows = cols
	const unsigned int matrixDimension = 64;

	const unsigned int bytesPerMatrix = matrixDimension * matrixDimension * sizeof(float);
	// Input matrix, output matrix
	const unsigned int matricesNeededPerCalculation = 2;
	// Max calculations in one batch
	const float maxMatrixCalculations = static_cast<float>(prop.totalGlobalMem) / (matricesNeededPerCalculation * bytesPerMatrix);
	std::cout << "Max calculations in one batch: " << maxMatrixCalculations << "\n";

	const unsigned int numMatrices = 20000;

	cublasHandle_t cuBlasHandle = nullptr;
	cublasSafeCall(cublasCreate(&cuBlasHandle));

	std::pair<float*, float**> devRandomInvertibleMatricesDesc = GpuCreatePositiveInvertibleSquareMatrices(cuBlasHandle, matrixDimension, numMatrices);

	int* devPivotArray; 
	gpuErrchk(cudaMalloc((void**)&devPivotArray, matrixDimension * numMatrices * sizeof(int)));

	int* devInfoArray;  
	gpuErrchk(cudaMalloc((void**)&devInfoArray, numMatrices * sizeof(int)));

	cublasSafeCall(
		cublasSgetrfBatched(
			cuBlasHandle, 
			matrixDimension, 
			devRandomInvertibleMatricesDesc.second,
			matrixDimension, 
			devPivotArray, 
			devInfoArray, 
			numMatrices));

	int* infoArray = (int*) malloc(numMatrices * sizeof(int));

	gpuErrchk(cudaMemcpy(infoArray, devInfoArray, numMatrices * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < numMatrices; ++i)
	{
		if (infoArray[i] != 0) 
		{
			std::cerr << "Factorization of matrix " << i << " failed: Matrix may be singular\n";
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

	// --- Allocate host space for the inverted matrices 
	float *invertedMatrices = new float[matrixDimension * matrixDimension * numMatrices];

	// --- Allocate device space for the inverted matrices 
	float *devInvertedMatrices; 
	gpuErrchk(cudaMalloc((void**)&devInvertedMatrices, matrixDimension * matrixDimension * numMatrices * sizeof(float)));

	// --- Creating the array of pointers needed as output to the batched getri
	float **invertedMatrixPointers = (float **)malloc(numMatrices * sizeof(float *));
	for (int i = 0; i < numMatrices; ++i)
	{
		invertedMatrixPointers[i] = (float *)((char*)devInvertedMatrices + i*((size_t)matrixDimension*matrixDimension) * sizeof(float));
	}

	float **devInvertedMatrixPointers;
	gpuErrchk(cudaMalloc((void**)&devInvertedMatrixPointers, numMatrices * sizeof(float *)));
	gpuErrchk(cudaMemcpy(devInvertedMatrixPointers, invertedMatrixPointers, numMatrices * sizeof(float *), cudaMemcpyHostToDevice));
	free(invertedMatrixPointers);

	cublasSafeCall(
		cublasSgetriBatched(
			cuBlasHandle, 
			matrixDimension, 
			(const float **)devRandomInvertibleMatricesDesc.second,
			matrixDimension,
			devPivotArray,
			devInvertedMatrixPointers, 
			matrixDimension,
			devInfoArray,
			numMatrices));

	gpuErrchk(cudaMemcpy(infoArray, devInfoArray, numMatrices * sizeof(int), cudaMemcpyDeviceToHost));

	for (int i = 0; i < numMatrices; ++i)
	{
		if (infoArray[i] != 0) 
		{
			std::cerr << "Inversion of matrix " << i << " failed: Matrix may be singular\n";
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

	gpuErrchk(
		cudaMemcpy(
			invertedMatrices, 
			devInvertedMatrices, 
			matrixDimension * matrixDimension * sizeof(float), 
			cudaMemcpyDeviceToHost));

	free(infoArray);
	free(invertedMatrices);
	gpuErrchk(cudaFree(devInfoArray));
	gpuErrchk(cudaFree(devPivotArray));
	gpuErrchk(cudaFree(devRandomInvertibleMatricesDesc.first));
	gpuErrchk(cudaFree(devRandomInvertibleMatricesDesc.second));
	gpuErrchk(cudaFree(devInvertedMatrices));
	gpuErrchk(cudaFree(devInvertedMatrixPointers));


	//const unsigned int blockDimX = 32;
	//const unsigned int blockDimY = 32;

	//dim3 blockDim(blockDimX, blockDimY);
	//dim3 gridDim((matrixDimension + blockDimX - 1) / blockDimX, (matrixDimension + blockDimY - 1) / blockDimY);
	//GpuInitIdentity <<<gridDim, blockDim >>>(thrust::raw_pointer_cast(&devI[0]), numberRowsI, numberColsI);

	//cusolverDnHandle_t cuSolverHandle = nullptr;
	//cusolverStatus_t cuSolverStatus = cusolverDnCreate(&cuSolverHandle);

	//cuSolverStatus = cusolverDnDestroy(cuSolverHandle);
	cublasSafeCall(cublasDestroy(cuBlasHandle));

    return 0;
}

