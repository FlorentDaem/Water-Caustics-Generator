## Résolution numérique de l'équation d'onde à 2D

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Lx = 10
Nx = 10
dx = Lx/Nx # voir pour avoir un nombre rond

Ly = Lx
Ny = Nx
dy = Ly/Ny

vals_x = np.array([i*dx for i in range(Nx)])
vals_y = np.array([j*dy for j in range(Ny)])

dt = 0.1

u = np.zeros([Nx, Ny])
old_u = np.zeros([Nx, Ny])

old_u[5,5] = 1
u[5,5] = 1


def prochains_u():
    next_u = np.zeros([Nx, Ny])

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            next_u[i, j] = 2*u[i, j] - old_u[i, j] + (dt/dx)**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j]) + (dt/dy)**2 * (u[i, j+1] - 2*u[i, j] + u[i, j-1])

    old_u[:,:] = u[:,:]
    u[:,:] = next_u[:,:]



# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')

# Définition d'une surface d'eau
grille_Z = u[:,:]

def affiche(grille_Z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, grille_Z, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    plt.show()

affiche(grille_Z)

def affiche_prochain():
    prochains_u()
    affiche(u[:,:])

for t in range(10):
    affiche_prochain()
