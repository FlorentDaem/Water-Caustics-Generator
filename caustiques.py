
## Imports

from tqdm import tqdm

from raytracing import *
from affichages import *
from surface import *


def calcul_trajectoires(rayons, surface, t, amplitude_fourier_plus, amplitude_fourier_moins):
    """
    Calcule les trajectoires de chaque rayon en utilisant les méthodes appropriées.

    Parameters
    ----------
    rayons : list
        Liste d'objets rayons.
    surface : array
        Surface d'eau.
    t : float
        Temps.
    amplitude_fourier_plus : array
        Amplitude de Fourier selon +omega.
    amplitude_fourier_moins : array
        Amplitude de Fourier selon -omega.
    """
    vecteurs_normaux = vecteurs_normaux_surface(surface)
    for i in tqdm(range(Nx-1), desc="Calcul des trajectoires "):
        for j in range(Ny-1):

            rayon = rayons[i][j]

            rayon.find_point_intersection(
                rayon, surface, vecteurs_normaux, "source")

            i, j = indices_du_point(rayon.point_interface)
            vecteur_normal = vecteurs_normaux[i, j]

            rayon.refract(vecteur_normal)

            rayon.find_point_intersection(
                rayon, surface, vecteurs_normaux, "surface")


## Animation

frames = 25
dt = 1/20


def genere_animation(surface, amplitude_fourier_plus, rayons, save_surface=True, save_motif=False):

    amplitude_fourier_moins = np.zeros((Nx, Ny), dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            amplitude_fourier_moins[i, j] = np.conjugate(
                amplitude_fourier_plus[-i, -j])

    for n in tqdm(range(frames), desc="frame"):
        if save_surface:
            save_frame_surface(surface, n)
        if save_motif:
            calcul_trajectoires(rayons, surface, n*dt,
                                amplitude_fourier_plus, amplitude_fourier_moins)
            save_image(surface, rayons, n)
        update_surface(surface, n*dt, amplitude_fourier_plus,
                       amplitude_fourier_moins)
    
    if save_surface:
        images = [Image.open(f"Frames/frame{n}.png") for n in range(frames)]
        images[0].save(f"gif/wave {Nx=}.gif", save_all=True,
                       append_images=images[1:], duration=dt*10**3, loop=0)
    if save_motif:
        images = [Image.open(f"Frames/frame{n} image.png")
                  for n in range(frames)]
        images[0].save(f"gif/caustiques dynamique {Nx=}.gif", save_all=True,
                       append_images=images[1:], duration=dt*10**3, loop=0)


