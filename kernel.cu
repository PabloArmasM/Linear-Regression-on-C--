

#include <cusolverDn.h>
#include <stdio.h>



#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <mkl.h>
#include <assert.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "eTimer.h"

using namespace std;

double resolveMKL(int m, int n, double *Y, double *Z) {
	int const x = 20, y = 5;
	//Lectura del fichero
	int result;

	double *YCopy = (double*)mkl_malloc(m*n * sizeof(double), 64);

	double *ZCopy = (double*)mkl_malloc(m * sizeof(double), 64);

	double *D = (double*)mkl_malloc(m * sizeof(double), 64);

	double *A = (double*)mkl_malloc(m * sizeof(double), 64);

	memcpy(YCopy, Y, m * n * sizeof(double));
	memcpy(ZCopy, Z, m * sizeof(double));

	//Resoulucion mediante el uso de las librerias LAPACK y BLAS
	result = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, 1, YCopy, n, ZCopy, 1);
	for (int i = 0; i < n; i++) {
		A[i] = ZCopy[i];
	}

	cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, -1, Y, n, A, 1, 1, Z, 1);
	//cblas_dgemv(CblasRowMajor, CblasTrans, m, 1, 1, Z, m, Z, 1, 0, Z, 1);

	double dot = cblas_ddot(m, Z, 1, Z, 1);
	dot = sqrt(dot);
	//printf("\n\n E: %1f\n\n", dot);


	mkl_free(YCopy);
	mkl_free(ZCopy);
	mkl_free(D);
	mkl_free(A);

	return dot;

}


double resolveCuda(int m, int n, double *Y, double *Z) {
	int lda, ldb, nrhs;
	lda = m;
	ldb = m;
	nrhs = 1;

	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	double *A = (double*)mkl_malloc(n * sizeof(double), 64);
	double *X = (double*)mkl_malloc(m * sizeof(double), 64);
	double *D = (double*)mkl_malloc(m * sizeof(double), 64);
	double *E = (double*)mkl_malloc(1 * sizeof(double), 64);

	double *Dev_Y = NULL;
	double *Dev_TAU = NULL;
	double*Dev_Z = NULL;
	double *Dev_A = NULL;
	int *Dev_Info = NULL;
	double *Dev_Work = NULL;
	double *Dev_E;
	int lwork = 0;

	int info_gpu = 0;


	const double one = 1;
	double inicio, fin = dsecnd();
	double timeCharge, timeSolve = 0.0;


	cusolver_status = cusolverDnCreate(&cusolverH);

	cublas_status = cublasCreate(&cublasH);

	cudaMalloc((void**)&Dev_Y, sizeof(double) * m*n);
	cudaMalloc((void**)&Dev_TAU, sizeof(double)*m*n);
	cudaMalloc((void**)&Dev_Z, sizeof(double)*m);
	cudaMalloc((void**)&Dev_Info, sizeof(int));
	cudaMalloc((void**)&Dev_A, sizeof(double)*n);
	cudaMalloc((void**)&Dev_E, sizeof(double)*m);
	//inicio = dsecnd();
	cudaMemcpy(Dev_Y, Y, sizeof(double)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Z, Z, sizeof(double)*m, cudaMemcpyHostToDevice);
	//fin = dsecnd();
	//timeCharge = fin - inicio;
	//Resolucion mediante el metodo de factorizacion QR, primero calculamos el tamaño del buffer
	//inicio = dsecnd();
	cusolver_status = cusolverDnDgeqrf_bufferSize(cusolverH, m, n, Dev_Y, lda, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	//fin = dsecnd();
	//timeSolve += fin - inicio;
	//inicio = dsecnd();
	cudaMalloc((void**)&Dev_Work, sizeof(double)*lwork);
	//fin = dsecnd();
	//timeCharge += fin - inicio;
	//Aqui calculamos la Q siendo esto d_A = Q*R
	//inicio = dsecnd();
	cusolver_status = cusolverDnDgeqrf(cusolverH, m, n, Dev_Y, lda, Dev_TAU, Dev_Work, lwork, Dev_Info);

	cudaDeviceSynchronize();


	//Con esto  lo que se calcula es Q^T*B
	cusolver_status = cusolverDnDormqr(
		cusolverH,
		CUBLAS_SIDE_LEFT,
		CUBLAS_OP_T,
		m, 1, n, Dev_Y, lda, Dev_TAU, Dev_Z, ldb, Dev_Work, lwork, Dev_Info);

	cudaDeviceSynchronize();


	//Esto da X = R / Q^T*B
	cublas_status = cublasDtrsm(
		cublasH,
		CUBLAS_SIDE_LEFT,
		CUBLAS_FILL_MODE_UPPER,
		CUBLAS_OP_N,
		CUBLAS_DIAG_NON_UNIT,
		n,
		1,
		&one, Dev_Y,
		lda,
		Dev_Z, ldb
	);
	cudaDeviceSynchronize();
	//fin = dsecnd();
	//timeSolve += fin - inicio;

	//inicio = dsecnd();
	cudaMemcpy(Dev_A, Dev_Z, sizeof(double)*n, cudaMemcpyDeviceToDevice);
	cudaMemcpy(A, Dev_Z, sizeof(double)*n, cudaMemcpyDeviceToHost);

	cudaMemcpy(Dev_Y, Y, sizeof(double)*m*n, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Z, Z, sizeof(double)*m, cudaMemcpyHostToDevice);
	//fin = dsecnd();
	//timeCharge += fin - inicio;
	///COMIENZO CUBLAS AGRESIVO////

	const double alpha = -1;
	const double beta = 1;
	//inicio = dsecnd();
	cublasDgemv(cublasH, CUBLAS_OP_N, m, n, &alpha, Dev_Y, lda, Dev_A, 1, &beta, Dev_Z, 1);
	//fin = dsecnd();
	//timeSolve += fin - inicio;

	//inicio = dsecnd();
	cudaMemcpy(Z, Dev_Z, sizeof(double)*m, cudaMemcpyDeviceToHost);
	//fin = dsecnd();
	//timeCharge += fin - inicio;

	//inicio = dsecnd();
	cublasDdot(cublasH, m, Dev_Z, 1, Dev_Z, 1, E);

	//cudaMemcpy(Z, Dev_Z, sizeof(double)*m, cudaMemcpyDeviceToHost);
	double Esqrt = sqrt(*E);
	//fin = dsecnd();
	//timeSolve += fin - inicio;
	/*printf("\n\n RESULTADO FINAL\n\n");
	printf("%1f", Esqrt);
	printf("\n\n Fin resultado final \n\n");*/


	//printf("\n\nTiempo dedicado a la transferencia de los datos: %1f msg", timeCharge*1.0e3);
	//printf("\nTiempo dedicado a la resolucion del problema: %1f msg\n", timeSolve*1.0e3);

	cudaFree(Dev_Y);
	cudaFree(Dev_Z);
	cudaFree(Dev_TAU);
	cudaFree(Dev_Info);
	cudaFree(Dev_Work);
	cudaFree(Dev_A);
	cudaFree(Dev_E);

	cublasDestroy(cublasH);
	cusolverDnDestroy(cusolverH);

	cudaDeviceReset();

	MKL_free(X);
	MKL_free(A);
	MKL_free(D);
	MKL_free(E);

	return Esqrt;


}

int main()
{

	printf("Ejercicio realizado por Pablo Armas Martin\n");
	//Lectura del fichero
	int m, n;


	std::ifstream fileOpen("9");
	fileOpen >> n >> m;
	n++;
	printf("Valor de m: %1i, valor de n: %1i", m, n);

	double *Y = (double*)mkl_malloc(m*n * sizeof(double), 64);
	double *YRow = (double*)mkl_malloc(m*n * sizeof(double), 64);
	double *Z = (double*)mkl_malloc(m * sizeof(double), 64);
	double *ZCopy = (double*)mkl_malloc(m * sizeof(double), 64);
	double *YCopy = (double*)mkl_malloc(m*n * sizeof(double), 64);

	double inicio, fin = dsecnd();
	double resultado;


	//Insertamos los datos en la matriz que utilizaremos posteriormente.
	//Imprimimos para comprobar que los datos son correctos.
	//La primera columna hace referencia a los valores que son necesarios para el vector X
	/*

	SE CONDIFICA COMO COLUMN MAJOR


	*/
	for (int i = 0; i < m; i++) {
		for (int j = 0; j <= n; j++) {
			if (j == 0) {
				fileOpen >> Z[i];
			}
			else if (j < n) {
				fileOpen >> Y[m*(j - 1) + i];
				YRow[i*n + (j - 1)] = Y[m*(j - 1) + i];
			}
			else {
				Y[m*(j - 1) + i] = 1.0;
				YRow[i*n + (j - 1)] = 1.0;
			}

		}
	}




	printf("\n\n Ejecucion codigo MKL\n\n");
	inicio = dsecnd();
	for (int i = 0; i < 5; i++){
	memcpy(YCopy, YRow, m * n * sizeof(double));
	memcpy(ZCopy, Z, m * sizeof(double));
	resultado = resolveMKL(m, n, YCopy, ZCopy);
	}
	fin = dsecnd();
	printf("Tiempo en CPU = %1f", ((fin - inicio) / 5.0)*1.0e3);
	printf("\nCon resultado: %1f\n", resultado);


	printf("\n\n Ejecucion codigo CUDA\n\n");

	inicio = dsecnd();
	for (int i = 0; i < 5; i++) {
		memcpy(YCopy, Y, m * n * sizeof(double));
		memcpy(ZCopy, Z, m * sizeof(double));
		resultado = resolveCuda(m, n, YCopy, ZCopy);
	}
	fin = dsecnd();
	printf("Tiempo en GPU = %1f msg", ((fin - inicio)/5.0)*1.0e3);
	printf("\nCon resultado: %1f\n", resultado);


	MKL_free(Y);
	MKL_free(Z);
	MKL_free(YCopy);
	MKL_free(ZCopy);
	MKL_free(YRow);


	std::getchar();
}
