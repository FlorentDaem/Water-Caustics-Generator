## Constantes

## Imports

import numpy as np


# Indice optique
n1 = 1
n2 = 1.3

# Constante de gravité
g = 9.81

# Constante capillaire
kc = 500

# Discrétisation de l'espace
Lx = 4
Nx = 2**9
dx = Lx/Nx

Ly = Lx
Ny = Nx
dy = Ly/Ny

vals_x = np.array([i*dx for i in range(Nx)])
vals_y = np.array([j*dy for j in range(Ny)])

# Discrétisation de l'espace dans le domaine de Fourier
dkx = 2*np.pi/Lx
dky = 2*np.pi/Ly


vecteurs_k = np.zeros((Nx, Ny, 2))
for i in range(Nx):
    for j in range(Ny):
        vecteurs_k[i, j] = np.array([(i-Nx/2)*dkx, (j-Ny/2)*dky])

# Profondeur d'eau
h = 3

if h < Lx:
    Lz = Lx
else:
    Lz = 2*h


# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')


# Facteur nécessaire pour changer de variables pour la FFT
fact_1 = np.ones((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        fact_1[i, j] = (-1)**(i+j)