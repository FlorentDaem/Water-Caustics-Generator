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

# Coefficient d'extinction
c = 0.5

# Discrétisation de l'espace
Lx = 1
Nx = 2**7
dx = Lx/Nx

Ly = Lx
Ny = Nx
dy = Ly/Ny

vals_x = np.array([i*dx for i in range(Nx)])
vals_y = np.array([j*dy for j in range(Ny)])

# Discrétisation de l'espace dans le domaine de Fourier
dkx = 2*np.pi/Lx
dky = 2*np.pi/Ly


vecteurs_kx = np.array([(i-Nx/2)*dkx for i in range(Nx)])
vecteurs_ky = np.array([(j-Ny/2)*dky for j in range(Ny)])


# Profondeur d'eau
H = 2

if H < Lx:
    Lz = Lx
else:
    Lz = 2*H


# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')


# Facteur nécessaire pour changer de variables pour la FFT
facteur_shift = np.ones((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        facteur_shift[i, j] = (-1)**(i+j)
