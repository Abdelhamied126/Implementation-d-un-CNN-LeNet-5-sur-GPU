#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

void MatrixInit(float *M, int n, int p) {
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

    // Parcours de chaque élément de la matrice
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            // Générer une valeur aléatoire entre -1 et 1
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%5.1f ", M[i * p + j]);
        }
        printf("\n");
    }
}


void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Kernel CUDA pour l'addition de matrices
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Ligne
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Colonne

    if (i < n && j < p) {
        int index = i * p + j; // Index de l'élément
        Mout[index] = M1[index] + M2[index];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    // Initialiser la matrice de sortie à 0
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0.0f;
        }
    }

    // Multiplication classique (3 boucles imbriquées)
    for (int i = 0; i < n; i++) { // Parcourir les lignes de M1
        for (int j = 0; j < n; j++) { // Parcourir les colonnes de M2
            for (int k = 0; k < n; k++) { // Parcourir les colonnes de M1 et les lignes de M2
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    // Calcul de l'index des threads (ligne et colonne)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Vérifier si l'index est dans les limites de la matrice
    if (row < n && col < n) {
        float sum = 0;

        // Multiplication ligne-colonne
        for (int k = 0; k < n; k++) {
            sum += M1[row * n + k] * M2[k * n + col];
        }

        // Stocker le résultat dans Mout
        Mout[row * n + col] = sum;
    }
}

int main() {
    int n = 10000, p = 10000; // Dimensions des matrices
    printf("n = %d, p = %d\n", n, p);
    size_t size = n * p * sizeof(float);

    // Allocation mémoire sur le CPU
    float *M1, *M2, *Mout_cpu, *Mout_gpu;
    M1 = (float*)malloc(size);
    M2 = (float*)malloc(size);
    Mout_cpu = (float*)malloc(size);
    Mout_gpu = (float*)malloc(size);

    // Initialisation des matrices
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Affichage des matrices initiales
    printf("Matrice M1 :\n");
    //MatrixPrint(M1, n, p);
    printf("\nMatrice M2 :\n");
    //MatrixPrint(M2, n, p);

    /*
    //////////////////////
    // ADDITION - CPU   //
    //////////////////////
    clock_t start_cpu_add = clock();
    MatrixAdd(M1, M2, Mout_cpu, n, p);
    clock_t end_cpu_add = clock();
    printf("\nRésultat addition CPU :\n");
    //MatrixPrint(Mout_cpu, n, p);
    printf("Temps addition CPU : %.4f secondes\n", (double)(end_cpu_add - start_cpu_add) / CLOCKS_PER_SEC);

     */

    //////////////////////
    // ADDITION - GPU   //
    //////////////////////
    float *d_M1, *d_M2, *d_Mout;
    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    printf("blockDim = %d %d\n ", blockDim.x, blockDim.y);

    //dim3 gridDim(1,1);
	dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (p + blockDim.y - 1) / blockDim.y);
    printf("grid dim: %d %d\n", gridDim.x, gridDim.y);

    cudaMatrixAdd<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n, p);

    cudaMemcpy(Mout_gpu, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nRésultat addition GPU :\n");
    //MatrixPrint(Mout_gpu, n, p);

	/*
    //////////////////////////
    // MULTIPLICATION - CPU //
    //////////////////////////
    clock_t start_cpu_mult = clock();
    MatrixMult(M1, M2, Mout_cpu, n);
    clock_t end_cpu_mult = clock();
    printf("\nRésultat multiplication CPU :\n");
    //MatrixPrint(Mout_cpu, n, p);
    printf("Temps multiplication CPU : %.4f secondes\n", (double)(end_cpu_mult - start_cpu_mult) / CLOCKS_PER_SEC);

	 */

    //////////////////////////
    // MULTIPLICATION - GPU //
    //////////////////////////
    cudaMatrixMult<<<gridDim, blockDim>>>(d_M1, d_M2, d_Mout, n);

    cudaMemcpy(Mout_gpu, d_Mout, size, cudaMemcpyDeviceToHost);

    printf("\nRésultat multiplication GPU :\n");
    //MatrixPrint(Mout_gpu, n, p);


    // Libération mémoire
    free(M1);
    free(M2);
    free(Mout_cpu);
    free(Mout_gpu);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}