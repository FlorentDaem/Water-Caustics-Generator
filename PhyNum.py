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
import noise

import mpl_toolkits.mplot3d.axes3d as p3

## Constantes

n1 = 1
n2 = 1.5

Lx = 10
Nx = 20
dx = Lx/Nx # voir pour avoir un nombre rond

Ly = Lx
Ny = Nx
dy = Ly/Ny

Lz = Lx

h = Lx/2  # faire varier la profondeur d'eau va jouer sur les motifs
A = Lx/100 # Amplitude des vagues
kx, ky = (2*np.pi/Lx, 2*np.pi/Ly)  # Vecteurs d'onde

vals_x = np.array([i*dx for i in range(Nx+1)])
vals_y = np.array([j*dy for j in range(Ny+1)])

# Définition d'un meshgrid
grille_X, grille_Y = np.meshgrid(vals_x, vals_y, indexing='ij')

dkx = 1/Nx
dky = 1/Ny

g = 9.81
kc = 1


## Fonctions

def vec(A, B):
    '''Renvoie le vecteur AB'''
    return B-A

def projection(v, n):
    '''Renvoie la composante de v qui est selon n'''
    return np.dot(v, n) * n

def symetrie(v, n):
    '''Revoie le symétrique de v par rapport à n'''
    return 2*projection(v,n) - v

def reflect(v, n):
    '''Renvoie la direction du rayon réfléchi'''
    return -symetrie(v, n)

def cos_theta_refract(cos_theta1):
    return np.sqrt(1-(n1/n2)**2 * (1-cos_theta1**2))

def refract(v, n):
    '''Renvoie la direction du rayon réfracté'''
    cos_theta1 = -np.dot(v, n)
    r = n1/n2*v + (n1/n2*cos_theta1- cos_theta_refract(cos_theta1))*n
    r = 1/np.linalg.norm(r)*r
    return r

def vecteurs_de_surface(surface):
    '''Renvoie les vecteurs normaux à la surface surface.'''
    vecteurs_normaux =[]
    for i in range(Nx+1-1):
        vecteurs_normaux.append([])
        for j in range(Ny+1-1):
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
    '''Renvoie le point qui correspond au rayon étendu à une distance s'''
    P, vec = rayon
    return P + s*vec


def indices_du_point(P):
    '''Renvoie les indices du pixel qui correspond au point P'''
    i = int(np.dot(P, np.array([1, 0, 0]))/dx)
    j = int(np.dot(P, np.array([0, 1, 0]))/dy)
    return (i, j)





def test_intersection(rayon, surface, s, vecteurs_normaux, intersection='sol'):
    '''Renvoie 0 si le rayon à la distance s appartient à la surface (au pixel) d'eau corespondant'''

    I = point_rayon(rayon, s)
    i, j = indices_du_point(I)

    if intersection == 'sol':
        P = np.zeros(3)
        n = np.array([0, 0, 1])
    
    elif intersection == 'surface':
        P = np.array([i*dx, j*dx, surface[i%Nx, j%Ny]])
        n = vecteurs_normaux[i%Nx, j%Ny]
    
    return np.dot(n, I-P)


def find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='sol'):
    '''Renvoie le point d'intersection du rayon avec la surface d'eau.
    On fait une recherche de zéro à l'aide de la fonction test_intersection.'''
    recherche_zero = scipy.optimize.root_scalar(lambda s: test_intersection(
        rayon, surface, s, vecteurs_normaux, intersection=intersection), x0=0, x1=Lz)
    s_intersection = recherche_zero.root
    I = point_rayon(rayon, s_intersection)
    return I


def calcul_trajectoires(rayons, surface):
    '''Renvoie les trajectoires de chaque rayon. C'est à dire l'ensemble de trois points (L, I, S),
    où L est le point de départ, I l'intersection avec l'eau, S l'intersection avec le sol.'''
    trajectoires = []
    vecteurs_normaux = vecteurs_de_surface(surface)
    for i in tqdm(range(Nx+1-1), desc="Calcul des trajectoires "):
        for j in range(Ny+1-1):
            rayon = rayons[i][j]
            L, u = rayon
            I = find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='surface')

            '''TODO : Faire en sorte de calculer chaque n dans find_point_intersection. Renvoyer I ET n.'''
            i, j = indices_du_point(I)
            n = vecteurs_normaux[i%Nx, j%Ny]

            v = refract(u, n)

            rayon = (I, v)

            S = find_point_intersection(rayon, surface, vecteurs_normaux, test_intersection, intersection='sol')

            trajectoires.append([L, I, S])
    return trajectoires


def calcul_motifs(trajectoires):
    motif = np.zeros((Nx+1, Ny+1))

    for trajectoire in trajectoires:
        L, I, S = trajectoire
        i_S, j_S = indices_du_point(S)

        # if (0 <= i_S and i_S < Nx-1) and (0 <= j_S and j_S < Ny-1):
        #     motif[i_S, j_S] += 1

        motif[i_S%Nx, j_S%Ny] += 1

    max_I = motif.max()
    # règle l'intensité de la lumière en fonction du nombre d'impacts de rayons
    motif[:, :] = motif[:, :]/max_I
    return motif

def motif_to_alpha(motif):
    image = np.zeros((Nx+1, Ny+1, 4))
    for i in range(Nx+1):
        for j in range(Ny+1):
            val = motif[i, j]
            alpha = val
            image[i,j] = [val, val, val, alpha]
    return image

    
def affiche_rayons(trajectoires, surface, save=False):
    '''Dessine les rayons et l'eau dans le plan y=0.'''
    for trajectoire in trajectoires:
        L, I, S = trajectoire
        iL, jL = indices_du_point(L)
        # iS, jS = indices_du_point(S)

        if jL == 0:
            plt.plot([L[0], I[0], S[0]], [L[2], I[2], S[2]], color='green')

    plt.plot(vals_x, surface[:,0])

    plt.xlim(0, Lx)
    plt.ylim(0, Lz)
    if save:
        plt.savefig("rayons.png")









def plot_surface(surface, save=False, n=None):
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
    if save:
        fig.savefig(f"Frames/frame{n}.png")
        plt.close(fig)






frames = 40
dt = 1/24





def omega(kx, ky):
    k = np.sqrt(kx**2 + ky**2)
    return k*g*(1+(k/kc)**2)**(1/2)





def surface_simple(u, t, A, B):
    for ix in range(Nx+1):
        for jy in range(Ny+1):
            
            x = ix*dx
            y = jy*dy

            u[ix, jy] = h

            sum = 0
            for ikx in range(0, Nx+1):
                for jky in range(0, Ny+1):

                    kx = ikx*dkx
                    ky = jky*dky
                    w = omega(kx, ky)

                    sum += A[ikx, jky]*np.exp(1j*(kx*x + ky*y - w*t)) + B[ikx, jky]*np.exp(1j*(kx*x + ky*y + w*t))
            
            u[ix, jy] += np.real(sum)/Nx/Ny

def genere_animation_simple(u, du0, du1):

    Fdu0 = np.fft.fft2(du0)
    Fdu1 = np.fft.fft2(du1)
    A = np.zeros((Nx+1, Ny+1), dtype=complex)
    B = np.zeros((Nx+1, Ny+1), dtype=complex)
    for ikx in range(0, Nx+1):
        for jky in range(0, Ny+1):
            w = omega(kx, ky)
            A[ikx, jky] = (np.exp(1j*w*dt)*Fdu0[ikx, jky] - Fdu1[ikx, jky])/(2j*np.sin(w*dt))
            B[ikx, jky] = (-np.exp(-1j*w*dt)*Fdu0[ikx, jky] + Fdu1[ikx, jky])/(2j*np.sin(w*dt))
    
    for n in tqdm(range(frames)):
        plot_surface(u, save=True, n=n)
        surface_simple(u, n*dt, A, B)
