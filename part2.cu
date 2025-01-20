#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>


// Initialisation d'une matrice avec des valeurs aléatoires entre 0 et 1
void initializeRandomMatrix(float *matrix, int size) {
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

    for (int i = 0; i < size; i++) {
        matrix[i] = ((float) rand()) / RAND_MAX; // Génère une valeur entre 0 et 1
    }
}

// Initialisation d'une matrice à 0
void initializeZeroMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f; // Initialise à 0
    }
}

// Fonction pour afficher une matrice 2D à partir d'un tableau 1D
void printMatrix2D(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%5.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

// Fonction pour afficher une matrice 3D à partir d'un tableau 1D
void printMatrix3D(float *matrix, int depth, int rows, int cols) {
    for (int d = 0; d < depth; d++) {
        printf("Depth %d:\n", d);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%5.2f ", matrix[d * rows * cols + i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void convolution2D(float *input, float *kernel, float *output,
                   int input_size, int kernel_size, int output_depth, int output_size) {
    int padding = 0; // Pas de padding
    int stride = 1;  // Stride de 1

    // Parcourir chaque canal de sortie (6 dans votre cas)
    for (int k = 0; k < output_depth; k++) {
        // Pour chaque position (i, j) dans l'image de sortie
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < output_size; j++) {
                float sum = 0.0f;

                // Convolution 2D avec le noyau k
                for (int m = 0; m < kernel_size; m++) {
                    for (int n = 0; n < kernel_size; n++) {
                        // Indices dans l'image d'entrée
                        int input_row = i * stride + m - padding;
                        int input_col = j * stride + n - padding;

                        // S'assurer que les indices sont valides
                        if (input_row >= 0 && input_row < input_size &&
                            input_col >= 0 && input_col < input_size) {
                            sum += input[input_row * input_size + input_col] *
                                   kernel[k * kernel_size * kernel_size + m * kernel_size + n];
                            }
                    }
                }

                // Stocker la valeur dans la matrice de sortie
                output[k * output_size * output_size + i * output_size + j] = sum;
            }
        }
    }
}

int main() {
    srand(time(NULL)); // Initialisation de la seed pour les valeurs aléatoires

    // Dimensions des matrices
    int raw_data_size = 32 * 32;
    int C1_data_size = 6 * 28 * 28;
    int S1_data_size = 6 * 14 * 14;
    int C1_kernel_size = 6 * 5 * 5;

    // Allocation mémoire pour les matrices
    float *raw_data = (float *)malloc(raw_data_size * sizeof(float));
    float *C1_data = (float *)malloc(C1_data_size * sizeof(float));
    float *S1_data = (float *)malloc(S1_data_size * sizeof(float));
    float *C1_kernel = (float *)malloc(C1_kernel_size * sizeof(float));

    // Initialisation des matrices
    initializeRandomMatrix(raw_data, raw_data_size);
    initializeRandomMatrix(C1_kernel, C1_kernel_size);
    initializeZeroMatrix(C1_data, C1_data_size);
    initializeZeroMatrix(S1_data, S1_data_size);

    // Affichage des matrices pour vérifier
    printf("Matrix raw_data (32x32):\n");
    printMatrix2D(raw_data, 32, 32);
    printf("\n");

    printf("Matrix C1_kernel (6x5x5):\n");
    printMatrix3D(C1_kernel, 6, 5, 5);
    printf("\n");

    printf("Matrix C1_data (6x28x28):\n");
    printMatrix3D(C1_data, 6, 28, 28);
    printf("\n");

    printf("Matrix S1_data (6x14x14):\n");
    printMatrix3D(S1_data, 6, 14, 14);
    printf("\n");

    // Libération mémoire
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}