### Projet PhyNum - Motifs au fond d'une piscine

# Florent Daem - M1 Physique fondamentale

## Imports

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import random as rd

import mpl_toolkits.mplot3d.axes3d as p3

## Constantes

n1 = 1
n2 = 1.3


Lx = 4
Nx = 2**5
dx = Lx/Nx # voir pour avoir un nombre rond

Ly = Lx
Ny = Nx
dy = Ly/Ny


h = 3  # faire varier la profondeur d'eau va jouer sur les motifs

if h < Lx:
    Lz = Lx
else :
    Lz = 2*h

vals_x = np.array([i*dx for i in range(Nx)])
vals_y = np.array([j*dy for j in range(Ny)])

# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')

dkx = 2*np.pi/Lx
dky = 2*np.pi/Ly


vecteurs_k = np.zeros((Nx, Ny, 2))
for i in range(Nx):
    for j in range(Ny):
        vecteurs_k[i,j] = np.array([(i-Nx/2)*dkx, (j-Ny/2)*dky])

g = 9.81
kc = 500

fact_1 = np.ones((Nx, Ny))
for i in range(Nx):
    for j in range(Ny):
        fact_1[i,j] = (-1)**(i+j)


## Fonctions

def vec(A, B):
    """
    Renvoie le vecteur AB.
    Les points A et B doivent avoir le même nombre de coordonnées.

    Parameters
    ----------
    A : Array numpy
        Coordonnées du point A
    B : Array numpy
        Coordonnées du point B

    Returns
    -------
    Array numpy
        Vecteur AB
    """
    return B-A

def projection(v, n):
    """
    Projette le vecteur v sur le vecteur n.

    Parameters
    ----------
    v : Array numpy
        Coordonnées du vecteur à projeter
    n : Array numpy
        Coordonnées du vecteur sur lequel on veut projeter

    Returns
    -------
    Array numpy
        Coordonnées du vecteur projection de v sur n
    """
    return np.dot(v, n) * n

def symetrie(v, n):
    """
    Revoie le vecteur symétrique de v par rapport à l'axe défini par n.

    Parameters
    ----------
    v : Array numpy
        Coordonnées du vecteur initial
    n : Array numpy
        Coordonnées du vecteur définissant l'axe

    Returns
    -------
    Array numpy
        Coordonnées du vecteur symétrique
    """
    return 2*projection(v,n) - v

def reflect(v, n):
    """
    Renvoie la direction du rayon v une fois réfléchi sur une surface de normale n.

    Parameters
    ----------
    v : Array numpy
        Coordonnées du rayon initial
    n : Array numpy
        Coordonnées du vecteur normal

    Returns
    -------
    Array numpy
        Coordonnées du rayon réfléchi
    """
    return -symetrie(v, n)

def cos_theta_refract(cos_theta1):
    return np.sqrt(1-(n1/n2)**2 * (1-cos_theta1**2))

def refract(ri, n):
    """
    Renvoie la direction du rayon ri une fois réfracté sur une surface de normale n.

    Parameters
    ----------
    ri : Array numpy
        Coordonnées du rayon initial
    n : Array numpy
        Coordonnées du vecteur normal

    Returns
    -------
    Array numpy
        Coordonnées du rayon réfracté
    """
    cos_theta1 = -np.dot(ri, n)
    rr = n1/n2*ri + (n1/n2*cos_theta1- cos_theta_refract(cos_theta1))*n
    rr = 1/np.linalg.norm(rr)*rr
    return rr

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


def point_rayon(rayon, s):
    """
    Renvoie le point qui correspond au rayon étendu à une distance s.

    Parameters
    ----------
    rayon : Array
        Rayon lumineux [P, vec, lum] partant de P, dirigé selon vec et de luminosité lum
    s : float
        Distance

    Returns
    -------
    Array numpy
        Coordonnées du point d'arrivée
    """
    P, vec, lum = rayon
    return P + s*vec


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
    intersection : str
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
    '''Renvoie le point d'intersection du rayon avec la surface d'eau.
    On fait une recherche de zéro à l'aide de la fonction test_intersection.'''
    recherche_zero = scipy.optimize.root_scalar(lambda s: test_intersection(
        rayon, surface, s, vecteurs_normaux, intersection), x0=0, x1=Lz)
    s_intersection = recherche_zero.root
    I = point_rayon(rayon, s_intersection)
    return I


def intensitee_refract(ri, n):
    theta_i = np.arccos(-np.dot(ri, n))
    theta_r = np.arcsin(n1/n2*np.sin(theta_i))
    return 1/2*((np.sin(theta_r-theta_i)**2)/(np.sin(theta_i+theta_r)**2) + (np.tan(theta_r-theta_i)**2)/(np.tan(theta_i+theta_r)**2))

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


def calcul_motifs(trajectoires):
    motif = np.zeros((Nx, Ny))

    for trajectoire in trajectoires:
        L, I, S, lum = trajectoire
        i_S, j_S = indices_du_point(S)

        # if (0 <= i_S and i_S < Nx-1) and (0 <= j_S and j_S < Ny-1):
        #     motif[i_S, j_S] += 1

        motif[i_S%Nx, j_S%Ny] += lum

    max_I = motif.max()
    # règle l'intensité de la lumière en fonction du nombre d'impacts de rayons
    motif[:, :] = motif[:, :]/max_I
    return motif

def motif_to_alpha(motif):
    image = np.zeros((Nx, Ny, 4))
    for i in range(Nx):
        for j in range(Ny):
            val = motif[i, j]
            alpha = val
            image[i,j] = [val, val, val, alpha]
    return image

    
def affiche_rayons(trajectoires, surface, save=False):
    '''Dessine les rayons et l'eau dans le plan y=0.'''
    for trajectoire in trajectoires:
        L, I, S, lum = trajectoire
        iL, jL = indices_du_point(L)
        # iS, jS = indices_du_point(S)

        if jL == 0:
            plt.plot([L[0], I[0], S[0]], [L[2], I[2], S[2]], color='green')

    plt.plot(vals_x, surface[:,0])

    plt.xlim(-Lx/2*0, Lx/2*2)
    plt.ylim(0, Lz)
    if save:
        plt.savefig("rayons.pdf")





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







def plot_surface(surface, save=False, n=None, fact=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-Lx/2*0, Lx/2*2)
    ax.set_ylim(-Ly/2*0, Ly/2*2)
    ax.set_zlim(0, Lz)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(grille_X, grille_Y, (surface-h)*fact + h, cmap="Blues",
                    linewidth=0, antialiased=False, alpha=0.9)
    if save:
        fig.savefig(f"Frames/frame{n}.png")
        plt.close(fig)


def save_image(surface, rayons, A, B, save=True, n=None):
    trajectoires = calcul_trajectoires(rayons, surface, A, B, n*dt)
    motif = calcul_motifs(trajectoires)

    motif = np.sqrt(motif)

    image = motif_to_alpha(motif)
    plt.imsave(f"Frames/frame {n} image.png", image)




frames = 25
dt = 1/10





def omega(kx, ky):
    k = np.sqrt(kx**2 + ky**2)
    return np.sqrt(k*g*(1+(k/kc)**2)*np.tanh(k*h))
    # return np.sqrt(k*g*(1+(k/kc)**2))
    # return np.sqrt(k*g)


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

def surface_simple(u, t, A, B):
    u[:, :] = h + fact_1[:,:] *np.real(np.fft.ifft2(surface_fourier(A, B, t)[:, :]))


def genere_animation_simple(u, h0, rayons, save_surface=True, save_motif=False):

    A = np.zeros((Nx, Ny), dtype=complex)
    B = np.zeros((Nx, Ny), dtype=complex)

    for i in range(0, Nx):
        for j in range(0, Ny):

            A[i, j] = h0[i, j]
            B[i, j] = np.conjugate(h0[-i+(Nx-1)*0, -j+(Nx-1)*0])
    
    for n in tqdm(range(frames), desc="frame"):
        if save_surface:
            plot_surface(u, save=True, n=n)
        if save_motif:
            save_image(u, rayons, A, B, n=n)
        surface_simple(u, n*dt, A, B)
