## Imports

import matplotlib.pyplot as plt

from constantes import *


def indices_du_point(P):
    """
    Renvoie les indices du pixel qui correspond au point P

    Parameters
    ----------
    P : Array numpy
        Coordonnées du point P

    Returns
    -------
    (int, int)
        Indices i et j tels que (i*dx, j*dy) = P
    """
    i = int(np.dot(P, np.array([1, 0, 0]))/dx)
    j = int(np.dot(P, np.array([0, 1, 0]))/dy)
    return (i, j)


def affiche_rayons(trajectoires, surface, save=False):
    '''Dessine les rayons et l'eau dans le plan y=0.'''
    for trajectoire in trajectoires:
        L, I, S, lum = trajectoire
        iL, jL = indices_du_point(L)
        # iS, jS = indices_du_point(S)

        if jL == 0:
            plt.plot([L[0], I[0], S[0]], [L[2], I[2], S[2]], color='green')

    plt.plot(vals_x, surface[:, 0])

    plt.xlim(-Lx/2*0, Lx/2*2)
    plt.ylim(0, Lz)
    if save:
        plt.savefig("images/rayons.pdf")


def plot_surface(surface, fact=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-Lx/2*0, Lx/2*2)
    ax.set_ylim(-Ly/2*0, Ly/2*2)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, (surface-H)*fact + H, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    plt.close(fig)


def save_frame_surface(surface, n, fact=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-Lx/2*0, Lx/2*2)
    ax.set_ylim(-Ly/2*0, Ly/2*2)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, (surface-H)*fact + H, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    fig.savefig(f"frames/frame{n}.png")
    plt.close(fig)


def calcul_motifs(trajectoires):
    motif = np.zeros((Nx, Ny))

    for trajectoire in trajectoires:
        L, I, S, lum = trajectoire
        i_S, j_S = indices_du_point(S)

        if (0 <= i_S and i_S < Nx-1) and (0 <= j_S and j_S < Ny-1):
            motif[i_S, j_S] += lum

    max_I = motif.max()
    # règle l'intensité de la lumière en fonction du nombre d'impacts de rayons
    motif[:, :] = motif[:, :]/max_I
    return motif


def motif_to_alpha(motif):
    """
    Convertion des valeurs de motif en alpha.

    Parameters
    ----------
    motif : array 2D
        Grille de valeurs de luminosité

    Returns
    -------
    array 2D
        Image avec valeurs de alpha
    """
    image = np.zeros((Nx, Ny, 4))
    for i in range(Nx):
        for j in range(Ny):
            val = motif[i, j]
            alpha = val
            image[i, j] = [val, val, val, alpha]
    return image


def save_image(surface, trajectoires, n):
    motif = calcul_motifs(trajectoires)

    motif = np.sqrt(motif)

    image = motif_to_alpha(motif)
    plt.imsave(f"frames/frame {n} image.png", image)

