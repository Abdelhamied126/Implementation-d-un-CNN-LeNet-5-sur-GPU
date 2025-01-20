# TP : Implémentation d'un CNN - LeNet-5 sur GPU

## Présentation du TP

Ce TP a pour objectif de vous familiariser avec l'utilisation de CUDA pour l'implémentation d'un réseau de neurones convolutif (CNN) classique, LeNet-5, proposé par Yann LeCun et al. en 1998. LeNet-5 est un réseau de neurones couramment utilisé pour la reconnaissance de chiffres manuscrits. L'objectif principal de ce TP est de comprendre et d'implémenter l'inférence (la phase de prédiction) de ce réseau sur un GPU en utilisant CUDA.

### Objectifs du TP

1. **Apprendre à utiliser CUDA** : Vous allez découvrir comment écrire des programmes en CUDA pour exploiter la puissance de calcul des GPU.
2. **Étudier la complexité des algorithmes** : Vous analyserez la complexité des opérations matricielles et comparerez les performances entre CPU et GPU.
3. **Implémenter un CNN** : Vous allez implémenter "from scratch" l'inférence du réseau LeNet-5, en vous concentrant sur les couches de convolution, de sous-échantillonnage, et les fonctions d'activation.
4. **Exporter et importer des données** : Vous apprendrez à exporter des données depuis un notebook Python et à les réimporter dans un projet CUDA.


### Structure du TP

Le TP est divisé en plusieurs parties :

1. **Prise en main de CUDA** : Vous commencerez par implémenter des opérations de base sur les matrices (addition, multiplication) sur CPU et GPU, et comparerez les performances.
2. **Implémentation des premières couches de LeNet-5** : Vous implémenterez les couches de convolution 2D et de sous-échantillonnage, ainsi que les fonctions d'activation.
3. **Intégration avec Python** : Vous utiliserez un notebook Jupyter pour entraîner le réseau et comprendre les dernières couches à implémenter en CUDA.
4. **Importation des données MNIST** : Vous chargerez les données MNIST et les utiliserez pour tester votre implémentation de LeNet-5.
5. **Exportation des poids** : Vous exporterez les poids du réseau entraîné en Python pour les utiliser dans votre programme CUDA.

### Fichiers fournis

- **LeNet5.ipynb** : Notebook Python contenant l'implémentation de LeNet-5 pour l'entraînement.
- **train-images.zip** : Fichier contenant les images d'entraînement du dataset MNIST.
- **printMNIST.cu** : Programme CUDA pour afficher une image du dataset MNIST.

### Outils recommandés

- **IDE** : Vous pouvez utiliser VS Code, CLion, ou Jupyter-Lab pour développer en CUDA. La compilation et l'exécution se feront via la console.

### Pour aller plus loin

Une fois l'inférence implémentée, vous pouvez explorer des pistes pour aller plus loin, comme l'entraînement du réseau de neurones sur GPU, ou l'optimisation de l'attention dans les réseaux de neurones modernes.

## Instructions pour l'utilisation

1. **Cloner le dépôt** : Utilisez `git clone` pour récupérer le projet sur votre machine.
2. **Exécuter les notebooks** : Ouvrez le notebook `LeNet5.ipynb` pour comprendre l'entraînement du réseau et exporter les poids.
3. **Implémenter en CUDA** : Utilisez les fichiers `.cu` pour implémenter les différentes couches de LeNet-5 en CUDA.
4. **Tester avec MNIST** : Chargez les données MNIST et testez votre implémentation en comparant les résultats avec ceux du notebook.

## Conclusion

Ce TP vous permettra de comprendre les bases de l'implémentation d'un réseau de neurones convolutif sur GPU en utilisant CUDA. Vous apprendrez à optimiser les opérations matricielles, à implémenter des couches de convolution, et à gérer des données complexes comme celles du dataset MNIST.

#### Pour plus de détails consulter sur les résultats :  
[Rapport.pdf](./Rapport.pdf)
