### Projet PhyNum - Motifs au fond d'une piscine

# Florent Daem - M1 Physique fondamentale

## Imports

from PIL import Image
import numpy as np
import scipy.optimize
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random as rd

import mpl_toolkits.mplot3d.axes3d as p3


from constantes import *
from raytracing import *
from affichages import *




## Fonctions



def vecteurs_de_surface(surface):
    """
    Renvoie les vecteurs normaux à la surface en calculant des produits vectoriels.

    Parameters
    ----------
    surface : Array (2D)
        Hauteur de la surface aux points (i,j)

    Returns
    -------
    Array (2D)
        Tableau des vecteurs normaux à la surface aux points (i,j)
    """
    vecteurs_normaux =[]
    for i in range(Nx-1):
        vecteurs_normaux.append([])
        for j in range(Ny-1):
            A = np.array([i*dx, j*dy, surface[i, j]])
            B = np.array([(i+1)*dx, j*dy, surface[i+1, j]])
            AB = vec(A, B)
            C = np.array([i*dx, (j+1)*dy, surface[i, j+1]])
            AC = vec(A, C)

            n = np.cross(AB, AC)
            n = 1/np.linalg.norm(n)*n
            vecteurs_normaux[i].append(n)
    return np.array(vecteurs_normaux)







def test_intersection(rayon, surface, s, vecteurs_normaux, interface):
    """
    Renvoie 0 si le rayon à la distance s appartient à l'interface.

    Parameters
    ----------
    rayon : Array
        Rayon lumineux [P, vec, lum] partant de P, dirigé selon vec et de luminosité lum
    surface : Array (2D)
        Hauteur de la surface aux points (i,j)
    s : float
        Distance
    vecteurs_normaux : Array (2D)
        Tableau des vecteurs normaux à la surface aux points (i,j)
    interface : str
        Nom de l'interface

    Returns
    -------
    float
        Nombre égal à 0 si et seulement si s*vec appartient à l'interface.
    """

    I = point_rayon(rayon, s)
    i, j = indices_du_point(I)

    if interface == 'sol':
        P = np.zeros(3)
        n = np.array([0, 0, 1])
    
    elif interface == 'surface':
        P = np.array([i*dx, j*dx, surface[i, j]])
        n = vecteurs_normaux[i, j]
    
    return np.dot(n, I-P)


def find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='sol'):
    """
    Renvoie le point d'intersection du rayon avec l'interface.
    On fait une recherche de zéro de la fonction test_intersection (en fonction de s).

    Parameters
    ----------
    rayon : Array
        Rayon lumineux [P, vec, lum] partant de P, dirigé selon vec et de luminosité lum
    surface : Array (2D)
        Hauteur de la surface aux points (i,j)
    vecteurs_normaux : Array (2D)
        Tableau des vecteurs normaux à la surface aux points (i,j)
    interface : str
        Nom de l'interface

    Returns
    -------
    Array
        Coordonnées du point d'intersection
    """
    recherche_zero = scipy.optimize.root_scalar(lambda s: test_intersection(
        rayon, surface, s, vecteurs_normaux, intersection), x0=0, x1=Lz)
    s_intersection = recherche_zero.root
    I = point_rayon(rayon, s_intersection)
    return I




def calcul_trajectoires(rayons, surface, A, B, t):
    '''Renvoie les trajectoires de chaque rayon. C'est à dire l'ensemble de trois points (L, I, S),
    où L est le point de départ, I l'intersection avec l'eau, S l'intersection avec le sol.'''
    trajectoires = []
    # vecteurs_normaux = vecteurs_normaux_avec_fourier(A, B, t)
    vecteurs_normaux = vecteurs_de_surface(surface)
    for i in tqdm(range(Nx-1), desc="Calcul des trajectoires "):
        for j in range(Ny-1):
            rayon = rayons[i][j]
            L, u, lum = rayon
            I = find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='surface')

            i, j = indices_du_point(I)
            n = vecteurs_normaux[i, j]

            v = refract(u, n)

            R = intensitee_refract(u,n)
            # print(R)
            T = 1-R

            rayon = (I, v, T*lum)

            S = find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='sol')

            trajectoires.append([L, I, S, lum])
    return trajectoires




    






def Ph_Phillips(kx, ky, V=np.array([1, 0]), A=0.001, l=0.01):
    "Calcule le spectre de vagues de Phillips."

    k = np.array([kx, ky])
    V_norm = np.linalg.norm(V)
    k_norm = np.linalg.norm(k)

    if V_norm*k_norm == 0:
        return 0
    else:

        k_unit = 1/k_norm*k
        V_unit = 1/V_norm*V

        cos_facteur = np.dot(k_unit, V_unit)

        L = V_norm**2/g

        correction = np.exp(-k_norm**2 * l**2)

        return A*np.exp(-1/(L*k_norm)**2)/k_norm**4 * cos_facteur**2 * correction


def random_h0(kx, ky, Ph, V):
    "Calcule une surface initiale aléatoire de vagues dans le domaine de Fourier."
    e_r = rd.gauss(0, 1)
    e_i = rd.gauss(0, 1)
    return 1/np.sqrt(2) * (e_r + 1j*e_i) * np.sqrt(Ph(kx, ky))






def omega(kx, ky):
    k = np.sqrt(kx**2 + ky**2)
    return np.sqrt(k*g*(1+(k/kc)**2)*np.tanh(k*h))


OMEGA = np.zeros((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):

        kx = (i-Nx/2)*dkx
        ky = (j-Ny/2)*dky
        OMEGA[i, j] = omega(kx, ky)
        if OMEGA[i, j] == 0:
            OMEGA[i, j] = 1e-5

def vecteurs_normaux_avec_fourier(A, B, t):
    grad_x = np.real(np.fft.ifft(1j*vecteurs_k[:,:,0]*np.fft.ifft(surface_fourier(A, B, t)[:,:], axis=1)))
    grad_y = np.real(np.fft.ifft(1j*vecteurs_k[:,:,1]*np.fft.ifft(surface_fourier(A, B, t)[:,:], axis=0)))
    norms = np.zeros((Nx, Ny, 3))
    for i in range(Nx):
        for j in range(Ny):
            norms[i,j] = np.array([-grad_x[i,j], -grad_y[i,j], 1])/np.sqrt(1+grad_x[i,j]**2+grad_y[i,j]**2)

    return norms

def gradient_surface(A, B, t):
    grad_x = np.real(np.fft.ifft(1j*vecteurs_k[:,:,0]*np.fft.ifft(surface_fourier(A, B, t)[:,:], axis=1)))
    grad_y = np.real(np.fft.ifft(1j*vecteurs_k[:,:,1]*np.fft.ifft(surface_fourier(A, B, t)[:,:], axis=0)))
    grad = np.zeros((Nx, Ny, 3))
    for i in range(Nx):
        for j in range(Ny):
            grad[i,j] = np.array([grad_x[i,j], grad_y[i,j], 0])
    return grad

def surface_fourier(A, B, t):
    return  Nx*Ny*(A[:, :]*np.exp(
        1j*(- OMEGA[:, :]*t)) + B[:, :]*np.exp(1j*(+ OMEGA[:, :]*t)))

def genere_surface(surface, t, A, B):
    surface[:, :] = h + fact_1[:,:] *np.real(np.fft.ifft2(surface_fourier(A, B, t)[:, :]))


frames = 25
dt = 1/10




def genere_animation_simple(surface, h0, rayons, save_surface=True, save_motif=False):

    A = np.zeros((Nx, Ny), dtype=complex)
    B = np.zeros((Nx, Ny), dtype=complex)

    for i in range(Nx):
        for j in range(Ny):

            A[i, j] = h0[i, j]
            B[i, j] = np.conjugate(h0[-i+(Nx-1)*0, -j+(Nx-1)*0])

    for n in tqdm(range(frames), desc="frame"):
        if save_surface:
            plot_surface(surface, n)
        if save_motif:
            save_image(surface, calcul_trajectoires(rayons, surface, A, B, n*dt), n)
        genere_surface(surface, n*dt, A, B)
