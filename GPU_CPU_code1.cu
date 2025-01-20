#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h> 

#define RAW_SIZE 32
#define C1_SIZE 28
#define S1_SIZE 14
#define KERNEL_SIZE 5
#define NUM_KERNELS 6


// Fonction pour afficher une matrice
void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%0.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Fonction pour initialiser une matrice aléatoire avec des valeurs entre 0 et 1
void initMatrixRandomCPU(float *matrix, int size) {
    srand(time(NULL)); // Initialisation du générateur de nombres aléatoires
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / RAND_MAX; // Génère des valeurs entre 0 et 1
    }
}

// Fonction pour initialiser une matrice avec des zéros
void initMatrixZeroCPU(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f;
    }
}

// Kernel CUDA pour la convolution 2D
__global__ void convolution2D(float *raw_data, float *C1_kernel, float *C1_data, int raw_size, int kernel_size, int C1_size, int num_kernels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index de la ligne
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index de la colonne
    int kernel_idx = blockIdx.z; // Index du noyau (une grille par noyau)

    if (row < C1_size && col < C1_size) {
        float sum = 0.0f;

        // Parcourir le noyau de convolution
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int raw_row = row + i; // Position correspondante dans raw_data
                int raw_col = col + j;
                sum += raw_data[raw_row * raw_size + raw_col] * C1_kernel[kernel_idx * kernel_size * kernel_size + i * kernel_size + j];
            }
        }

        // Stocker le résultat
        C1_data[kernel_idx * C1_size * C1_size + row * C1_size + col] = sum;
    }
}


__global__ void subsampling2D(float *C1_data, float *S1_data, int C1_size, int S1_size, int num_kernels) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Index de la ligne
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index de la colonne
    int kernel_idx = blockIdx.z; // Index du noyau (une grille par noyau)

    if (row < S1_size && col < S1_size) {
        // Moyennage des 4 pixels (2x2 bloc)
        float sum = 0.0f;
        int base_row = row * 2;
        int base_col = col * 2;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                sum += C1_data[kernel_idx * C1_size * C1_size + (base_row + i) * C1_size + (base_col + j)];
            }
        }

        // Calcul de la moyenne et stockage
        S1_data[kernel_idx * S1_size * S1_size + row * S1_size + col] = sum / 4.0f;
    }
}



// Fonction d'activation tangente hyperbolique
__device__ float activation_tanh(float M) {
    float e_plus = expf(M);   // e^M
    float e_minus = expf(-M); // e^(-M)
    return (e_plus - e_minus) / (e_plus + e_minus); // tanh(M)
}

// Kernel pour appliquer la fonction d'activation tanh
__global__ void applyActivationTanh(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index global du thread

    if (idx < size) {
        output[idx] = activation_tanh(input[idx]); // Application de tanh
    }
}


int main() {

    float *d_COUCHE_activated;
    // Dimensions des matrices
    int raw_size = RAW_SIZE * RAW_SIZE;
    int c1_size = NUM_KERNELS * C1_SIZE * C1_SIZE;
    int s1_size = NUM_KERNELS * S1_SIZE * S1_SIZE;
    int kernel_size = NUM_KERNELS * KERNEL_SIZE * KERNEL_SIZE;

    // Allocation de mémoire sur le CPU
    float *h_raw_data = (float *)malloc(raw_size * sizeof(float));
    float *h_C1_data = (float *)malloc(c1_size * sizeof(float));
    float *h_S1_data = (float *)malloc(s1_size * sizeof(float)); 
    float *h_C1_kernel = (float *)malloc(kernel_size * sizeof(float));
    float *h_S1_kernel = (float *)malloc(kernel_size * sizeof(float));
    d_COUCHE_activated = (float *)malloc(s1_size * sizeof(float));
    float *h_C1_activated = (float *)malloc(s1_size * sizeof(float));
   

    /*

    // Vérification des allocations mémoire
    if (!h_raw_data || !h_C1_data || !h_S1_data || !h_C1_kernel) {
        fprintf(stderr, "Erreur : allocation de mémoire sur le CPU échouée.\n");
        return -1;
    }

     */

    // Initialisation des matrices
    initMatrixRandomCPU(h_raw_data, raw_size);    // Données d'entrée
    initMatrixZeroCPU(h_C1_data, c1_size);        // Résultat de la première convolution
    initMatrixZeroCPU(h_S1_data, s1_size);        // Résultat du sous-échantillonnage
    initMatrixRandomCPU(h_C1_kernel, kernel_size); // Noyaux de convolution

    // Allocation de mémoire sur le GPU
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data,*d_S1_kernel;
    cudaMalloc((void **)&d_raw_data, raw_size * sizeof(float));
    cudaMalloc((void **)&d_C1_data, c1_size * sizeof(float));
    cudaMalloc((void **)&d_C1_kernel, kernel_size * sizeof(float));
    cudaMalloc((void **)&d_S1_data, s1_size * sizeof(float));
    cudaMalloc((void **)&d_S1_kernel, kernel_size * sizeof(float));
    cudaMalloc((void **)&d_COUCHE_activated, c1_size * sizeof(float));    


    /*

    /*
    // Vérification des allocations mémoire sur le GPU
    if (!d_raw_data || !d_C1_data || !d_C1_kernel || !d_C1_data || !d_S1_data) {
        fprintf(stderr, "Erreur : allocation de mémoire sur le GPU échouée.\n");
        free(h_raw_data);
        free(h_C1_data);
        free(h_S1_data);
        free(h_C1_kernel);
        free(h_S1_data);
        free(h_S1_kernel);
        return -1;
    }

     */

    // Copie des données du CPU vers le GPU
    cudaMemcpy(d_raw_data, h_raw_data, raw_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, h_C1_data, c1_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, h_C1_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Définition des dimensions des blocs et grilles
    dim3 blockDim(16, 16); // Taille des blocs (16x16 threads)
    dim3 gridDim((C1_SIZE + blockDim.x - 1) / blockDim.x, (C1_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS); // Une grille par noyau

    dim3 blockDimActivation(256); // Taille des blocs
    dim3 gridDimActivation((c1_size + blockDimActivation.x - 1) / blockDimActivation.x); // Nombre de blocs



    
    /////////////////////////////////////CALCULS DES COUCHES/////////////////////////////////////////////////

    // Lancer le kernel de convolution
    convolution2D<<<gridDim, blockDim>>>(d_raw_data, d_C1_kernel, d_C1_data, RAW_SIZE, KERNEL_SIZE, C1_SIZE, NUM_KERNELS);
    cudaDeviceSynchronize();
    // Copier les résultats sur le CPU
    cudaMemcpy(h_C1_data, d_C1_data, c1_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Lancer le kernel de sous-échantillonnage
    subsampling2D<<<gridDim, blockDim>>>(d_C1_data, d_S1_data, C1_SIZE, S1_SIZE, NUM_KERNELS);
    cudaDeviceSynchronize();
    // Copier les résultats sur le CPU
    cudaMemcpy(h_S1_data, d_S1_data, s1_size * sizeof(float), cudaMemcpyDeviceToHost);


    //Exécution du kernel d'activation 
    applyActivationTanh<<<gridDimActivation, blockDimActivation>>>(d_S1_data, d_COUCHE_activated, c1_size);
    cudaDeviceSynchronize();
    // Copier les résultats sur le CPU
    cudaMemcpy(h_C1_activated, d_COUCHE_activated, c1_size * sizeof(float), cudaMemcpyDeviceToHost);

    ////////////////////////////////////////////////////////////////////////////////////////////




    // Affichage des résultats
    printf("Raw Data (32x32):\n");
    printMatrix(h_raw_data, RAW_SIZE, RAW_SIZE);

    printf("C1 Kernel (6x5x5):\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("Kernel %d:\n", k + 1);
        printMatrix(&h_C1_kernel[k * KERNEL_SIZE * KERNEL_SIZE], KERNEL_SIZE, KERNEL_SIZE);
    }

    printf("C1 Data (6x28x28):\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("Feature Map %d:\n", k + 1);
        printMatrix(&h_C1_data[k * C1_SIZE * C1_SIZE], C1_SIZE, C1_SIZE);
    }

    printf("C1 Activated Data (6x28x28):\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("Activated Feature Map %d:\n", k + 1);
        printMatrix(&h_C1_activated[k * C1_SIZE * C1_SIZE], C1_SIZE, C1_SIZE);
    }


    printf("S1 Data (6x14x14):\n");
    for (int k = 0; k < NUM_KERNELS; k++) {
        printf("Subsampled Feature Map %d:\n", k + 1);
        printMatrix(&h_S1_data[k * S1_SIZE * S1_SIZE], S1_SIZE, S1_SIZE);
    }

    // Libération de la mémoire CPU
    free(h_raw_data);
    free(h_C1_data);
    free(h_S1_data);
    free(h_C1_kernel);
    free(h_S1_data);
    free(h_S1_kernel);
    free(h_C1_activated);



    // Libération de la mémoire GPU
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_S1_data);
    cudaFree(d_S1_kernel);
    cudaFree(d_COUCHE_activated);

    return 0;
}
