#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 28
#define HEIGHT 28

unsigned int readBigEndianInt(FILE *file) {
    unsigned char bytes[4];
    fread(bytes, 1, 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}


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

// Fonction pour charger les poids à partir d'un fichier texte
void loadWeights(const char *filePath, float *weights, int size) {
    FILE *file = fopen(filePath, "r");  // Ouvrir le fichier en mode lecture
    if (file == NULL) {
        fprintf(stderr, "Erreur : Impossible d'ouvrir le fichier %s\n", filePath);
        exit(EXIT_FAILURE);
    }

    // Lire les poids ligne par ligne
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &weights[i]) != 1) {
            fprintf(stderr, "Erreur : Lecture échouée pour l'indice %d dans le fichier %s\n", i, filePath);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);  // Fermer le fichier après la lecture
}

void loadBiases(const char *filePath, float *destination, int size) {
    FILE *file = fopen(filePath, "r"); // Ouvrir en mode texte
    if (file == NULL) {
        fprintf(stderr, "Erreur : Impossible d'ouvrir le fichier %s\n", filePath);
        exit(EXIT_FAILURE);
    }

    // Lire les biais ligne par ligne
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &destination[i]) != 1) {
            fprintf(stderr, "Erreur : Lecture échouée pour l'indice %d dans le fichier %s\n", i, filePath);
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
	/*
    // Vérification des biais chargés
    printf("Biais chargés depuis %s :\n", filePath);
    for (int i = 0; i < size; i++) {
        printf("Biais[%d] = %f\n", i, destination[i]);
    }

	 */

    fclose(file); // Fermer le fichier après la lecture
}

__global__ void cudaDenseLayer(float *input, float *weights, float *bias, float *output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[idx * input_size + i]; // Matrice des poids
        }
        output[idx] = sum + bias[idx];  // Ajout du biais
    }
}


__global__ void cudaAveragePooling2D(float *input, float *output,
                                     int input_rows, int input_cols,
                                     int pool_size, int num_channels) {
    // Identification des indices globaux pour les threads
    int channel = blockIdx.z;          // Canal traité
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Ligne de la sortie
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonne de la sortie

    int output_rows = input_rows / pool_size; // Dimensions de la sortie
    int output_cols = input_cols / pool_size;

    // Vérifier les limites
    if (row < output_rows && col < output_cols && channel < num_channels) {
        // Accéder au canal actuel
        float sum = 0.0f;
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                // Index de l'entrée
                int input_row = row * pool_size + i;
                int input_col = col * pool_size + j;
                int input_index = channel * input_rows * input_cols
                                  + input_row * input_cols + input_col;

                // Accumuler la valeur
                sum += input[input_index];
            }
        }

        // Calculer la moyenne et stocker dans la sortie
        int output_index = channel * output_rows * output_cols
                           + row * output_cols + col;
        output[output_index] = sum / (pool_size * pool_size);
    }
}

__global__ void cudaActivationTanh(float *input, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    input[idx] = tanhf(input[idx]);
  }
}

__global__ void cudaSoftmax(float *input, float *output, int size) {
    // Index global du thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Réduction partagée pour calculer la somme des exponentielles
    __shared__ float partial_sum;

    // Initialisation de la somme partielle
    if (threadIdx.x == 0) {
        partial_sum = 0.0f;
    }
    __syncthreads();

    // Étape 1 : Calcul des exponentielles
    float exp_value = 0.0f;
    if (idx < size) {
        exp_value = expf(input[idx]);
        atomicAdd(&partial_sum, exp_value);  // Réduction atomique dans la mémoire partagée
    }
    __syncthreads();

    // Étape 2 : Normalisation (chaque thread met à jour son élément)
    if (idx < size) {
        output[idx] = exp_value / partial_sum;
    }
}
__global__ void cudaConvolution2D(float *input, float *kernels, float *output,
                                  int input_rows, int input_cols,
                                  int kernel_size, int output_rows, int output_cols,
                                  int num_kernels, float *biases) {
    // Identification des indices globaux pour chaque thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Ligne de l'image
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Colonne de l'image
    int kernel_idx = blockIdx.z; // Index du filtre (kernel)

    // Vérification des limites pour éviter les erreurs d'accès mémoire
    if (row < output_rows && col < output_cols && kernel_idx < num_kernels) {
        float sum = 0.0f;

        // Parcours des éléments du kernel (filtre)
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                // Calcul des indices pour l'image d'entrée
                int input_row = row + i;
                int input_col = col + j;

                // Accès à l'entrée et au kernel
                float input_val = input[input_row * input_cols + input_col];
                float kernel_val = kernels[kernel_idx * kernel_size * kernel_size + i * kernel_size + j];

                // Accumulation des produits
                sum += input_val * kernel_val;
            }
        }

        // Ajout du biais pour ce filtre (kernel)
        sum += biases[kernel_idx];

        // Sauvegarde du résultat dans la sortie
        output[kernel_idx * output_rows * output_cols + row * output_cols + col] = sum;
    }
}

// Sauvegarder l'image normale dans un fichier texte ou binaire pour visualisation
void saveImageToFile(float *image, const char *filename, int width, int height) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Erreur : Impossible de créer le fichier %s\n", filename);
        return;
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(file, "%.6f ", image[i * width + j]);  // Export des pixels normalisés
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("Image sauvegardée dans %s\n", filename);
}

int main() {
    // Allocation mémoire pour les données d'entrée et intermédiaires
    float *input_raw_data = (float*)malloc(28 * 28 * sizeof(float)); // Données d'entrée brute
    float *C1_output_data = (float*)malloc(6 * 28 * 28 * sizeof(float)); // Sortie de la couche C1 (Conv2D)
    float *S1_output_data = (float*)malloc(6 * 14 * 14 * sizeof(float)); // Sortie de la couche S1 (AveragePooling2D)
    float *C2_output_data = (float*)malloc(16 * 10 * 10 * sizeof(float)); // Sortie de la couche C2 (Conv2D)
    float *S2_output_data = (float*)malloc(16 * 5 * 5 * sizeof(float)); // Sortie de la couche S2 (AveragePooling2D)
    float *D1_output_data = (float*)malloc(120 * sizeof(float)); // Sortie de la couche Dense 1 (C5)
    float *D2_output_data = (float*)malloc(84 * sizeof(float)); // Sortie de la couche Dense 2 (F6)
    float *D3_output_data = (float*)malloc(10 * sizeof(float)); // Sortie de la couche Dense 3 (Output)

    // Allocation mémoire pour les kernels des couches de convolution
    float *C1_kernels = (float*)malloc(6 * 5 * 5 * sizeof(float)); // Noyaux 5x5 pour C1
    float *C2_kernels = (float*)malloc(16 * 5 * 5 * 6 * sizeof(float)); // Noyaux 5x5 pour C2

    // Allocation mémoire pour les noyaux de pooling
    float *average_pooling_kernel = (float*)malloc(2 * 2 * sizeof(float)); // Kernel 2x2 pour pooling

    // Allocation mémoire pour les biais
    float *C1_biases = (float*)malloc(6 * sizeof(float)); // Biais pour la couche C1
    float *C2_biases = (float*)malloc(16 * sizeof(float)); // Biais pour la couche C2
    float *D1_biases = (float*)malloc(120 * sizeof(float)); // Biais pour la couche Dense 1
    float *D2_biases = (float*)malloc(84 * sizeof(float)); // Biais pour la couche Dense 2
    float *D3_biases = (float*)malloc(10 * sizeof(float)); // Biais pour la couche Dense 3

    // Allocation mémoire pour les poids des couches denses
    float *D1_weights = (float*)malloc(400 * 120 * sizeof(float)); // Poids pour Dense 1 (C5)
    float *D2_weights = (float*)malloc(120 * 84 * sizeof(float)); // Poids pour Dense 2 (F6)
    float *D3_weights = (float*)malloc(84 * 10 * sizeof(float)); // Poids pour Dense 3 (Output)

    unsigned int magic, nbImg, nbRows, nbCols;
    FILE *fptr;

    // Allouer de la mémoire pour une image MNIST
    float *image = (float *)malloc(HEIGHT * WIDTH * sizeof(float));

    // Ouvrir le fichier IDX3
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Impossible d'ouvrir le fichier\n");
        exit(1);
    }

    // Lire l'en-tête du fichier
    magic = readBigEndianInt(fptr);
    nbImg = readBigEndianInt(fptr);
    nbRows = readBigEndianInt(fptr);
    nbCols = readBigEndianInt(fptr);

    printf("Magic Number : %u\n", magic);
    printf("Number of Images : %u\n", nbImg);
    printf("Number of Rows : %u\n", nbRows);
    printf("Number of Columns : %u\n", nbCols);

    if (nbRows != HEIGHT || nbCols != WIDTH) {
        printf("Dimensions inattendues des images MNIST\n");
        exit(1);
    }

    int image_index = 0; // Indice de l'image à charger (par défaut 0)
	printf("Indice de l'image chargée en CUDA : %d\n", image_index);

    // Se positionner à l'image souhaitée dans le fichier IDX3
	fseek(fptr, 16 + image_index * HEIGHT * WIDTH, SEEK_SET);

    // Lire une image MNIST et la charger dans input_raw_data
	for (int i = 0; i < HEIGHT; i++) {
    	for (int j = 0; j < WIDTH; j++) {
        	unsigned char pixel;
        	fread(&pixel, sizeof(unsigned char), 1, fptr);  // Lire un pixel
        	input_raw_data[i * WIDTH + j] = (float)pixel / 255.0f;  // Normaliser entre 0 et 1
    	}
	}

	// Sauvegarder l'image chargée
	saveImageToFile(input_raw_data, "cuda_loaded_image.txt", WIDTH, HEIGHT);

    // Charger les poids et biais depuis les fichiers
    loadWeights("weights_txt_output/layers_conv2d_vars_0.txt", C1_kernels, 6 * 5 * 5);
    loadBiases("weights_txt_output/layers_conv2d_vars_1.txt", C1_biases, 6);

    loadWeights("weights_txt_output/layers_conv2d_1_vars_0.txt", C2_kernels, 16 * 5 * 5 * 6);
    loadBiases("weights_txt_output/layers_conv2d_1_vars_1.txt", C2_biases, 16);

    loadWeights("weights_txt_output/layers_dense_vars_0.txt", D1_weights, 400 * 120);
    loadBiases("weights_txt_output/layers_dense_vars_1.txt", D1_biases, 120);

    loadWeights("weights_txt_output/layers_dense_1_vars_0.txt", D2_weights, 120 * 84);
    loadBiases("weights_txt_output/layers_dense_1_vars_1.txt", D2_biases, 84);

    loadWeights("weights_txt_output/layers_dense_2_vars_0.txt", D3_weights, 84 * 10);
    loadBiases("weights_txt_output/layers_dense_2_vars_1.txt", D3_biases, 10);

    // Initialisation du GPU
    float *d_input_raw_data, *d_C1_output_data, *d_S1_output_data, *d_C2_output_data;
    float *d_S2_output_data, *d_D1_output_data, *d_D2_output_data, *d_D3_output_data;
    float *d_C1_kernels, *d_C2_kernels, *d_D1_weights, *d_D2_weights, *d_D3_weights;
    float *d_C1_biases, *d_C2_biases, *d_D1_biases, *d_D2_biases, *d_D3_biases;

    // Allocation sur le GPU
    cudaMalloc((void**)&d_input_raw_data, 28 * 28 * sizeof(float));
    cudaMalloc((void**)&d_C1_output_data, 6 * 28 * 28 * sizeof(float));
    cudaMalloc((void**)&d_S1_output_data, 6 * 14 * 14 * sizeof(float));
    cudaMalloc((void**)&d_C2_output_data, 16 * 10 * 10 * sizeof(float));
    cudaMalloc((void**)&d_S2_output_data, 16 * 5 * 5 * sizeof(float));
    cudaMalloc((void**)&d_D1_output_data, 120 * sizeof(float));
    cudaMalloc((void**)&d_D2_output_data, 84 * sizeof(float));
    cudaMalloc((void**)&d_D3_output_data, 10 * sizeof(float));
    cudaMalloc((void**)&d_C1_kernels, 6 * 5 * 5 * sizeof(float));
    cudaMalloc((void**)&d_C2_kernels, 16 * 5 * 5 * 6 * sizeof(float));
    cudaMalloc((void**)&d_D1_weights, 400 * 120 * sizeof(float));
    cudaMalloc((void**)&d_D2_weights, 120 * 84 * sizeof(float));
    cudaMalloc((void**)&d_D3_weights, 84 * 10 * sizeof(float));
    cudaMalloc((void**)&d_C1_biases, 6 * sizeof(float));
    cudaMalloc((void**)&d_C2_biases, 16 * sizeof(float));
    cudaMalloc((void**)&d_D1_biases, 120 * sizeof(float));
    cudaMalloc((void**)&d_D2_biases, 84 * sizeof(float));
    cudaMalloc((void**)&d_D3_biases, 10 * sizeof(float));

    // Copier les données du CPU vers le GPU
    cudaMemcpy(d_input_raw_data, input_raw_data, 28 * 28 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernels, C1_kernels, 6 * 5 * 5 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_biases, C1_biases, 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_kernels, C2_kernels, 16 * 5 * 5 * 6 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_biases, C2_biases, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1_weights, D1_weights, 400 * 120 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1_biases, D1_biases, 120 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D2_weights, D2_weights, 120 * 84 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D2_biases, D2_biases, 84 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D3_weights, D3_weights, 84 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_D3_biases, D3_biases, 10 * sizeof(float), cudaMemcpyHostToDevice);

    printf("All weights and biases loaded and transferred to GPU.\n");

    printMatrix(input_raw_data, 28, 28);

    // Couche C1 : Convolution + Activation (Tanh)
   	dim3 threadsPerBlock_C1(16, 16); // Chaque bloc gère une région 16x16
	dim3 blocksPerGrid_C1((28 + 15) / 16, (28 + 15) / 16, 6); // Nombre de blocs nécessaires pour couvrir l'image
    cudaConvolution2D<<<blocksPerGrid_C1, threadsPerBlock_C1>>>(d_input_raw_data, d_C1_kernels, d_C1_output_data, 28, 28, 5, 28, 28, 6, d_C1_biases);
    cudaActivationTanh<<<6, 28 * 28>>>(d_C1_output_data, 6 * 28 * 28);

    float *C1_output_after_activation = (float *)malloc(6 * 28 * 28 * sizeof(float)); // Allocation pour 6 feature maps
	cudaMemcpy(C1_output_after_activation, d_C1_output_data, 6 * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < 6; i++) { // 6 feature maps
    	printf("Feature map %d (C1 after activation):\n", i);
    	printMatrix(&C1_output_after_activation[i * 28 * 28], 28, 28);
	}

    /*
    printf("Kernels pour C1 :\n");
	for (int k = 0; k < 6; k++) { // 6 kernels
    	printf("Kernel %d:\n", k + 1);
   		printMatrix(&C1_kernels[k * 5 * 5], 5, 5); // Chaque kernel est 5x5
	}

     */

    // Couche S1 : Pooling (AveragePooling2D)
	dim3 threadsPerBlockS1(8, 8);  // Chaque bloc traite une région de 8x8
	dim3 blocksPerGrid_S1((14 + 7) / 8, (14 + 7) / 8, 6);  // Une grille 3D : (largeur, hauteur, nombre de filtres)
    cudaAveragePooling2D<<<blocksPerGrid_S1, threadsPerBlockS1>>>(d_C1_output_data, d_S1_output_data, 28, 28, 2, 6);

    float *S1_output_host = (float*)malloc(6 * 14 * 14 * sizeof(float));
	cudaMemcpy(S1_output_host, d_S1_output_data, 6 * 14 * 14 * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 6; i++) {  // 6 filtres
	    printf("Feature map %d (S1 output):\n", i);
	    printMatrix(&S1_output_host[i * 14 * 14], 14, 14);  // Chaque carte est 14x14
	}

    // Couche C2 : Convolution + Activation (Tanh)
    dim3 threadsPerBlockC2(10, 10); // Nombre de threads par bloc pour C2
	dim3 blocksPerGrid_C2((10 + 9) / 10, (10 + 9) / 10, 16);
    cudaConvolution2D<<<blocksPerGrid_C2, threadsPerBlockC2>>>(d_S1_output_data, d_C2_kernels, d_C2_output_data, 14, 14, 5, 10, 10,16, d_C2_biases);
    cudaActivationTanh<<<16, 10 * 10>>>(d_C2_output_data, 16 * 10 * 10);

    float *C2_output_after_activation = (float *)malloc(16 * 10 * 10 * sizeof(float)); // Allocation pour 6 feature maps
	cudaMemcpy(C2_output_after_activation, d_C2_output_data, 16 * 10 * 10 * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < 6; i++) { // 6 feature maps
    	printf("Feature map %d (C2 after activation):\n", i);
    	printMatrix(&C2_output_after_activation[i * 10 * 10], 10, 10);
	}

    // Couche S2 : Pooling (AveragePooling2D)
    dim3 threadsPerBlockS2(5, 5); // Nombre de threads par bloc pour S2
	dim3 blocksPerGrid_S2((5 + 4) / 5, (5 + 4) / 5, 16);  // Grille pour couvrir tous les canaux
    cudaAveragePooling2D<<<blocksPerGrid_S2, threadsPerBlockS2>>>(d_C2_output_data, d_S2_output_data, 10, 10, 2, 16);

    float *S2_output_host = (float *)malloc(16 * 5 * 5 * sizeof(float));
	cudaMemcpy(S2_output_host, d_S2_output_data, 16 * 5 * 5 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 16; i++) {
    	printf("Feature map %d (S2 output):\n", i);
    	printMatrix(&S2_output_host[i * 5 * 5], 5, 5);
	}

    // Couche Dense 1 (C5) : Fully Connected + Bias + Activation (Tanh)
    cudaDenseLayer<<<1, 120>>>(d_S2_output_data, d_D1_weights, d_D1_biases, d_D1_output_data, 400, 120);
    cudaActivationTanh<<<1, 120>>>(d_D1_output_data, 120);

    float *D1_output_host = (float *)malloc(120 * sizeof(float));
	cudaMemcpy(D1_output_host, d_D1_output_data, 120 * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Sortie de Dense 1 (C5) avant activation :\n");
	for (int i = 0; i < 120; i++) {
	    printf("%0.6f ", D1_output_host[i]);
	}
	printf("\n");

    // Couche Dense 2 (F6) : Fully Connected + Bias + Activation (Tanh)
    cudaDenseLayer<<<1, 84>>>(d_D1_output_data, d_D2_weights, d_D2_biases, d_D2_output_data, 120, 84);
    cudaActivationTanh<<<1, 84>>>(d_D2_output_data, 84);

    float *D2_output_host = (float *)malloc(84 * sizeof(float));
	cudaMemcpy(D2_output_host, d_D2_output_data, 84 * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Sortie de Dense 2 (F6) avant activation :\n");
	for (int i = 0; i < 84; i++) {
	    printf("%0.6f ", D2_output_host[i]);
	}
	printf("\n");

    // Couche Dense 3 (Output) : Fully Connected + Softmax
    cudaDenseLayer<<<1, 10>>>(d_D2_output_data, d_D3_weights, d_D3_biases, d_D3_output_data, 84, 10);
    cudaSoftmax<<<1, 10>>>(d_D3_output_data, d_D3_output_data, 10);

    float *D3_output_host = (float *)malloc(10 * sizeof(float));
	cudaMemcpy(D3_output_host, d_D3_output_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Sortie de Dense 3 (Output) avant Softmax :\n");
	for (int i = 0; i < 10; i++) {
	    printf("%0.6f ", D3_output_host[i]);
	}
	printf("\n");

    // Copie des résultats du GPU vers le CPU
    cudaMemcpy(D3_output_data, d_D3_output_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage des résultats
    printf("Résultats finaux (Probabilités Softmax):\n");
    for (int i = 0; i < 10; i++) {
        printf("Classe %d : %f\n", i, D3_output_data[i]);
    }

    int predicted_class = 0;
	float max_prob = D3_output_data[0];
	for (int i = 1; i < 10; i++) {
	    if (D3_output_data[i] > max_prob) {
	        max_prob = D3_output_data[i];
	        predicted_class = i;
	    }
	}
	printf("Classe prédite : %d avec probabilité %f\n", predicted_class, max_prob);

    // Cleanup
    free(input_raw_data);
    free(C1_output_data);
    free(S1_output_data);
    free(C2_output_data);
    free(S2_output_data);
    free(D1_output_data);
    free(D2_output_data);
    free(D3_output_data);
    free(C1_kernels);
    free(C2_kernels);
    free(average_pooling_kernel);
    free(C1_biases);
    free(C2_biases);
    free(D1_biases);
    free(D2_biases);
    free(D3_biases);
    free(D1_weights);
    free(D2_weights);
    free(D3_weights);
    free(image);
    free(C1_output_after_activation);
    free(C2_output_after_activation);
    free(S1_output_host);
    free(D2_output_host);
    free(S2_output_host);
    free(D3_output_host);
    free(D1_output_host);



    cudaFree(d_input_raw_data);
    cudaFree(d_C1_output_data);
    cudaFree(d_S1_output_data);
    cudaFree(d_C2_output_data);
    cudaFree(d_S2_output_data);
    cudaFree(d_D1_output_data);
    cudaFree(d_D2_output_data);
    cudaFree(d_D3_output_data);
    cudaFree(d_C1_kernels);
    cudaFree(d_C2_kernels);
    cudaFree(d_D1_weights);
    cudaFree(d_D2_weights);
    cudaFree(d_D3_weights);
    cudaFree(d_C1_biases);
    cudaFree(d_C2_biases);
    cudaFree(d_D1_biases);
    cudaFree(d_D2_biases);
    cudaFree(d_D3_biases);

    return 0;
}