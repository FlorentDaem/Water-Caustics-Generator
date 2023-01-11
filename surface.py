
## Imports

from PIL import Image
import numpy as np

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random as rd

import mpl_toolkits.mplot3d.axes3d as p3


from constantes import *




def spectre_Phillips(kx, ky, V=np.array([10, 0]), A=0.001/Lx/Ly, l=Lx*10**-2):
    """
    Calcule le spectre de vagues de Phillips.

    Parameters
    ----------
    kx : float
        Composante x du vecteur d'onde.
    ky : float
        Composante y du vecteur d'onde.
    V : array 2D, optional
        Vecteur vitesse du vent, by default np.array([10, 0]).
    A : float, optional
        Amplitude du spectre, by default 0.001.
    l : float, optional
        Distance caractéristique d'aténuation, by default 0.01.

    Returns
    -------
    float
        Valeur du spectre de vagues de Phillips en (kx, ky).
    """

    k = np.array([kx, ky])
    V_norm = np.linalg.norm(V)
    k_norm = np.linalg.norm(k)

    if V_norm*k_norm == 0:
        return 0
    else:

        k_unit = 1/k_norm*k
        V_unit = 1/V_norm*V

        facteur_direction = np.dot(k_unit, V_unit)

        L = V_norm**2/g

        correction = np.exp(-k_norm**2 * l**2)

        return A*np.exp(-1/(L*k_norm)**2)/k_norm**4 * facteur_direction**2 * correction


def random_amplitude_fourier_plus(kx, ky, spectre):
    """
    Calcule une surface initiale aléatoire de vagues dans le domaine de Fourier.

    Parameters
    ----------
    kx : float
        Composante x du vecteur d'onde.
    ky : float
        Composante y du vecteur d'onde.
    spectre : fonction
        Spectre utilisé.

    Returns
    -------
    float
        Valeur du spectre en (kx, ky) avec un bruit gaussien.
    """
    e_r = rd.gauss(0, 1)
    e_i = rd.gauss(0, 1)
    return 1/np.sqrt(2) * (e_r + 1j*e_i) * np.sqrt(spectre(kx, ky, V))






def omega(kx, ky):
    """
    Relation de dispersion.

    Parameters
    ----------
    kx : float
        Composante x du vecteur d'onde.
    ky : float
        Composante y du vecteur d'onde.

    Returns
    -------
    float
        Valeur de la pulsation en (kx, ky).
    """
    k = np.sqrt(kx**2 + ky**2)
    return np.sqrt(k*g*(1+(k/kc)**2)*np.tanh(k*H))


# Calcule et enregistre les valeurs de la pulsation
OMEGA = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        OMEGA[i, j] = omega(vals_kx[i], vals_ky[j])


## Calcul de la surface d'eau

def surface_fourier(t, amplitude_fourier_plus, amplitude_fourier_moins):
    """
    Génère les composantes de Fourier de la surface au temps t.

    Parameters
    ----------
    t : float
        Temps.
    amplitude_fourier_plus : array
        Amplitude de Fourier selon +omega.
    amplitude_fourier_moins : array
        Amplitude de Fourier selon -omega.

    Returns
    -------
    array
        Surface dans l'espace de Fourier.
    """
    onde_plus = amplitude_fourier_plus[:, :]*np.exp(1j*(+ OMEGA[:, :]*t))
    onde_moins = amplitude_fourier_moins[:, :]*np.exp(1j*(- OMEGA[:, :]*t))
    return  (onde_moins + onde_plus)

def update_surface(surface, t, amplitude_fourier_plus, amplitude_fourier_moins):
    """
    Génère la surface à l'instant t avec une FFT inverse.

    Parameters
    ----------
    surface : float
        Surface à l'instant précédent.
    t : float
        Temps.
    amplitude_fourier_plus : array
        Amplitude de Fourier selon +omega.
    amplitude_fourier_moins : array
        Amplitude de Fourier selon -omega.
    """
    surface_dans_fourier = surface_fourier(t, amplitude_fourier_plus, amplitude_fourier_moins)[:, :]
    surface[:, :] = H + facteur_shift[:,:] *np.real(np.fft.ifft2(surface_dans_fourier, norm="forward"))



