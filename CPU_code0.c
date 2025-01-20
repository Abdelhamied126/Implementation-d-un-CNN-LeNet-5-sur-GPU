#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Fonction pour initialiser une matrice avec des valeurs aléatoires
void MatrixInit(float *M, int n, int p) {
    srand(time(NULL)); // Initialisation de la graine pour les nombres aléatoires
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            M[i * p + j] = (float)rand() / RAND_MAX; // Valeurs entre 0 et 1
        }
    }
}

// Fonction pour afficher une matrice
void MatrixPrint(float *M, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            printf("%.2f ", M[i * p + j]);
        }
        printf("\n");
    }
}

// Fonction pour additionner deux matrices : Mout = M1 + M2
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

// Exemple d'utilisation
int main() {
    int n = 3, p = 3; // Dimensions des matrices

    // Allocation mémoire pour les matrices
    float *M1 = (float *)malloc(n * p * sizeof(float));
    float *M2 = (float *)malloc(n * p * sizeof(float));
    float *Mout = (float *)malloc(n * p * sizeof(float));

    // Initialisation des matrices M1 et M2
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);

    // Affichage des matrices initialisées
    printf("Matrice M1 :\n");
    MatrixPrint(M1, n, p);

    printf("\nMatrice M2 :\n");
    MatrixPrint(M2, n, p);

    // Addition des matrices
    MatrixAdd(M1, M2, Mout, n, p);

    // Affichage du résultat de l'addition
    printf("\nMatrice Mout (M1 + M2) :\n");
    MatrixPrint(Mout, n, p);

    // Libération de la mémoire
    free(M1);
    free(M2);
    free(Mout);

    return 0;
}
