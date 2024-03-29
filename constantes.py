## Constantes

## Imports

import numpy as np


# Indice optique
n1 = 1
n2 = 1.3

# Constante de gravité
g = 9.81

# Constante capillaire
kc = 370

# Coefficient d'extinction
c = 0.5

# Discrétisation de l'espace
Lx = 1
Nx = 2**9
dx = Lx/Nx

Ly = Lx
Ny = Nx
dy = Ly/Ny

vals_x = np.array([i*dx for i in range(Nx)])
vals_y = np.array([j*dy for j in range(Ny)])

Nx_sol = Nx
dx_sol = Lx/Nx_sol

Ny_sol = Nx_sol
dy_sol = Ly/Ny_sol

# Décalage des rayons par rapport à la grille

shift_rayons = False

if shift_rayons :
    di_rayon = 1/2
    dj_rayon = 1/2
else :
    di_rayon = 0
    dj_rayon = 0

# Discrétisation de l'espace dans le domaine de Fourier
dkx = 2*np.pi/Lx
dky = 2*np.pi/Ly


vals_kx = np.array([(i-Nx/2)*dkx for i in range(Nx)])
vals_ky = np.array([(j-Ny/2)*dky for j in range(Ny)])


# Profondeur d'eau
H = 2

if H < Lx:
    Lz = Lx
else:
    Lz = 2*H

# Vitesse du vent
V = np.array([1,0]) * 1

# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')


# Facteur nécessaire pour changer de variables pour la FFT
facteur_shift = np.ones((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        facteur_shift[i, j] = (-1)**(i+j)
