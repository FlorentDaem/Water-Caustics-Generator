## Imports

import matplotlib.pyplot as plt
import os

from constantes import *


def indices_du_point(P):
    """
    Renvoie les indices du pixel qui correspond au point P.

    Parameters
    ----------
    P : array
        Coordonnées du point P.

    Returns
    -------
    (int, int)
        Indices i et j tels que (i*dx, j*dy) = P.
    """
    i = int(np.dot(P, np.array([1, 0, 0]))/dx)
    j = int(np.dot(P, np.array([0, 1, 0]))/dy)
    return (i, j)


def affiche_rayons(rayons, surface, save=False):
    """
    Dessine les rayons et l'eau dans le plan y=0.

    Parameters
    ----------
    rayons : list
        Liste d'objets rayon.
    surface : array
        Surface d'eau.
    save : bool, optional
        Enregistre la figure si save==True, by default False.
    """
    for i in range(Nx-1):
        for j in range(Ny-1):
            rayon = rayons[i][j]
            point_source = rayon.point_source
            point_interface = rayon.point_interface
            point_sol = rayon.point_sol
            i, j = indices_du_point(rayon.point_source)

            # Plot les rayons du plan y=0
            if j == 0:
                plt.plot([point_source[0], point_interface[0], point_sol[0]], [point_source[2], point_interface[2], point_sol[2]], color='red')

    plt.plot(vals_x, surface[:, 0])

    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    if save:
        nom_dossier = "images"
        nom_fichier = "rayons.pdf"
        plt.savefig(os.path.join(nom_dossier, nom_fichier))


def plot_surface(surface):
    """
    Plot la surface d'eau en 3D.

    Parameters
    ----------
    surface : array
        Surface d'eau.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, surface, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    plt.close(fig)


def save_frame_surface(surface, n, nom_dossier):
    """
    Enregistre une frame du plot de la surface d'eau en 3D.

    Parameters
    ----------
    surface : array
        Surface d'eau.
    n : int
        Numéro de la frame.
    nom_dossier : str
        Nom du dossier d'enregistrement.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, surface, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    nom_fichier = f"frame{n}.png"
    fig.savefig(os.path.join(nom_dossier, nom_fichier))
    plt.close(fig)


def calcul_motifs(rayons):
    """
    Détermine la luminosité du motif.

    Parameters
    ----------
    rayons : list
        Liste de rayons.

    Returns
    -------
    array
        Image du motif.
    """
    motif = np.zeros((Nx, Ny))

    for i in range(Nx-1):
        for j in range(Ny-1):
            rayon = rayons[i][j]
            point_sol = rayon.point_sol
            i_S, j_S = indices_du_point(point_sol)

            # Ajoute de la luminosité à chaque pixel dans l'intervalle considéré
            if (0 <= i_S and i_S < Nx-1) and (0 <= j_S and j_S < Ny-1):
                motif[i_S, j_S] += rayon.lum_sol

    max_I = motif.max()
    # Règle l'intensité de la lumière en fonction du nombre d'impacts de rayons
    motif[:, :] = motif[:, :]/max_I
    return motif


def motif_to_alpha(motif):
    """
    Convertion des valeurs de motif en alpha.

    Parameters
    ----------
    motif : array 2D
        Grille de valeurs de luminosité.

    Returns
    -------
    array 2D
        Image avec valeurs de alpha.
    """
    image = np.zeros((Nx, Ny, 4))
    for i in range(Nx):
        for j in range(Ny):
            val = motif[i, j]
            alpha = val
            image[i, j] = [val, val, val, alpha]
    return image


def save_image(surface, rayons, n, nom_dossier):
    """
    Sauvegarde une frame du motif.

    Parameters
    ----------
    surface : array
        Surface d'eau.
    rayons : list
        Liste d'objets rayons.
    n : int
        Numéro de la frame.
    nom_dossier : str
        Nom du dossier d'enregistrement.
    """
    motif = calcul_motifs(rayons)

    image = motif_to_alpha(motif)
    nom_fichier = f"frame{n} image.png"
    plt.imsave(os.path.join(nom_dossier, nom_fichier), image)
    plt.close()

