#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// *** Fonctions CPU ***

// Initialisation d'une matrice avec des valeurs aléatoires entre -1 et 1
void MatrixInit(float *M, int n, int p) {
    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

// Affichage d'une matrice
void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// Addition de deux matrices sur le CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Multiplication de deux matrices sur le CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Mout[i * n + j] = 0; // Initialisation
            for (int k = 0; k < n; k++) {
                Mout[i * n + j] += M1[i * n + k] * M2[k * n + j];
            }
        }
    }
}

// *** Fonctions GPU ***

// Kernel CUDA pour l'addition de matrices
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Ligne
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Colonne

    if (i < n && j < p) {
        int index = i * p + j; // Index de l'élément
        Mout[index] = M1[index] + M2[index];
    }
}

// Kernel CUDA pour la multiplication de matrices
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Ligne
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Colonne

    if (row < n && col < n) {
        float value = 0;
        for (int k = 0; k < n; k++) {
            value += M1[row * n + k] * M2[k * n + col];
        }
        Mout[row * n + col] = value;
    }
}

// *** Fonction principale ***

int main() {
    int n = 1000, p = 1000; // Dimensions des matrices
    size_t size = n * p * sizeof(float);

    // Allocation mémoire sur l'hôte (CPU)
    float *M1 = (float *)malloc(size);
    float *M2 = (float *)malloc(size);
    float *MoutCPU = (float *)malloc(size);
    float *MoutGPU = (float *)malloc(size);

    // Initialisation des matrices avec des valeurs aléatoires entre -1 et 1
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Affichage des matrices initiales
    printf("Matrice M1 :\n");
    //MatrixPrint(M1, n, p);
    printf("\nMatrice M2 :\n");
    //MatrixPrint(M2, n, p);

    // *** CUDA : Addition de matrices ***

    float *d_M1, *d_M2, *d_Mout; // Pointeurs pour le GPU

    // Allocation mémoire sur le GPU
    cudaMalloc((void **)&d_M1, size);
    cudaMalloc((void **)&d_M2, size);
    cudaMalloc((void **)&d_Mout, size);

    // Copie des données de l'hôte (CPU) vers le GPU
    cudaMemcpy(d_M1, M1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, size, cudaMemcpyHostToDevice);

    // Définition de la grille et des blocs pour l'addition
    dim3 threadsPerBlock(16, 16);
    //dim3 blocksPerGrid(1,1);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Lancement du kernel CUDA pour l'addition
    cudaMatrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, p);

    // Copie du résultat du GPU vers l'hôte
    cudaMemcpy(MoutGPU, d_Mout, size, cudaMemcpyDeviceToHost);

    // Affichage du résultat de l'addition CUDA
    printf("\nRésultat CUDA (M1 + M2) :\n");
    //MatrixPrint(MoutGPU, n, p);

    // *** CUDA : Multiplication de matrices ***

    // Lancement du kernel CUDA pour la multiplication
    cudaMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n);

    // Copie du résultat du GPU vers l'hôte
    cudaMemcpy(MoutGPU, d_Mout, size, cudaMemcpyDeviceToHost);

    // Affichage du résultat de la multiplication CUDA
    printf("\nRésultat CUDA (M1 * M2) :\n");
    //MatrixPrint(MoutGPU, n, p);
    
    

    clock_t start_cpu_add = clock();
    // *** CPU : Multiplication de matrices ***
    MatrixMult(M1, M2, MoutCPU, n);
     clock_t end_cpu_add = clock();
    // Affichage du résultat CPU
    printf("\nRésultat CPU (M1 * M2) :\n");
    //MatrixPrint(MoutCPU, n, p);
    printf("Temps addition CPU : %.4f secondes\n", (double)(end_cpu_add - start_cpu_add) / CLOCKS_PER_SEC);

    // Libération de la mémoire sur le GPU
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    // Libération de la mémoire sur l'hôte
    free(M1);
    free(M2);
    free(MoutCPU);
    free(MoutGPU);

    return 0;
}
