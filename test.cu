#include <stdio.h>
#include <stdlib.h>

#define HEIGHT 28  // Ajustez selon votre fichier IDX3
#define WIDTH 28

void charBckgrndPrint(char *str, int rgb[3]) {
    printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("%s\033[0m", str);
}

void imgColorPrint(int height, int width, int ***img) {
    int row, col;
    char *str = "  ";
    for (row = 0; row < height; row++) {
        for (col = 0; col < width; col++) {
            charBckgrndPrint(str, img[row][col]);
        }
        printf("\n");
    }
}

// Fonction pour lire un entier 32 bits en big-endian
unsigned int readBigEndianInt(FILE *fptr) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, fptr);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

int main() {
    int i, j;
    int ***img;
    int color[3] = {255, 0, 0}; // Couleur rouge
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;
    int imageIndex = 2; // Indice de l'image à lire (commence à 1)

    // Allouer la mémoire pour l'image
    img = (int ***)malloc(HEIGHT * sizeof(int **));
    for (i = 0; i < HEIGHT; i++) {
        img[i] = (int **)malloc(WIDTH * sizeof(int *));
        for (j = 0; j < WIDTH; j++) {
            img[i][j] = (int *)malloc(sizeof(int) * 3);
        }
    }

    // Ouvrir le fichier
    if ((fptr = fopen("train-images.idx3-ubyte", "rb")) == NULL) {
        printf("Impossible d'ouvrir le fichier");
        exit(1);
    }

    // Lire l'entête
    magic = readBigEndianInt(fptr);
    nbImg = readBigEndianInt(fptr);
    nbRows = readBigEndianInt(fptr);
    nbCols = readBigEndianInt(fptr);

    printf("Magic Number: %u\n", magic);
    printf("Number of Images: %u\n", nbImg);
    printf("Rows: %u, Columns: %u\n", nbRows, nbCols);

    if (imageIndex > nbImg) {
        printf("L'indice d'image dépasse le nombre d'images disponibles\n");
        exit(1);
    }

    // Positionner le pointeur sur l'image désirée
    fseek(fptr, (imageIndex - 1) * nbRows * nbCols, SEEK_CUR);

    // Lire l'image
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) { 
            fread(&val, sizeof(unsigned char), 1, fptr);
            img[i][j][0] = (int)val * color[0] / 255;
            img[i][j][1] = (int)val * color[1] / 255;
            img[i][j][2] = (int)val * color[2] / 255;
        }
    }

    // Afficher l'image
    imgColorPrint(HEIGHT, WIDTH, img);

    // Fermer le fichier et libérer la mémoire
    fclose(fptr);
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            free(img[i][j]);
        }
        free(img[i]);
    }
    free(img);

    return 0;
}

