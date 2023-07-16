
## Imports

import numpy as np
import scipy.optimize

from constantes import *


def vec(A, B):
    """
    Renvoie le vecteur AB.
    Les points A et B doivent avoir le même nombre de coordonnées.

    Parameters
    ----------
    A : array
        Coordonnées du point A.
    B : array
        Coordonnées du point B.

    Returns
    -------
    array
        Vecteur AB.
    """
    return B-A


def projection(v, n):
    """
    Projette le vecteur v sur le vecteur n.

    Parameters
    ----------
    v : array
        Coordonnées du vecteur à projeter.
    n : array
        Coordonnées du vecteur sur lequel on veut projeter.

    Returns
    -------
    array
        Coordonnées du vecteur projection de v sur n.
    """
    return np.dot(v, n) * n


def symetrie(v, n):
    """
    Revoie le vecteur symétrique de v par rapport à l'axe défini par n.

    Parameters
    ----------
    v : array
        Coordonnées du vecteur initial.
    n : array
        Coordonnées du vecteur définissant l'axe.

    Returns
    -------
    array
        Coordonnées du vecteur symétrique.
    """
    return 2*projection(v, n) - v


def reflect(v, n):
    """
    Renvoie la direction du rayon v une fois réfléchi sur une surface de normale n.

    Parameters
    ----------
    v : array
        Coordonnées du rayon initial.
    n : array
        Coordonnées du vecteur normal.

    Returns
    -------
    array
        Coordonnées du rayon réfléchi.
    """
    return -symetrie(v, n)


def cos_theta_refract(cos_theta_i):
    """
    Calcul de l'angle de réfraction en fonction de l'angle d'incidence.

    Parameters
    ----------
    cos_theta_i : float
        Angle d'incidence.

    Returns
    -------
    float
        Angle de réfraction.
    """
    return np.sqrt(1-(n1/n2)**2 * (1-cos_theta_i**2))


def refract(vecteur_direction_i, vecteur_normal):
    """
    Renvoie la direction du rayon vecteur_direction_i une fois réfracté sur une surface de normale vecteur_normal.

    Parameters
    ----------
    vecteur_direction_i : array
        Coordonnées du rayon initial.
    vecteur_normal : array
        Coordonnées du vecteur normal.

    Returns
    -------
    array
        Coordonnées du rayon réfracté.
    """
    cos_theta_i = -np.dot(vecteur_direction_i, vecteur_normal)
    rr = n1/n2*vecteur_direction_i + (n1/n2*cos_theta_i - cos_theta_refract(cos_theta_i))*vecteur_normal
    rr = 1/np.linalg.norm(rr)*rr
    return rr


def point_rayon(rayon, s):
    """
    Renvoie le point qui correspond au rayon étendu à une distance s.

    Parameters
    ----------
    rayon : array
        Rayon lumineux [P, vec, lum] partant de P, dirigé selon vec et de luminosité lum.
    s : float
        Distance.

    Returns
    -------
    array
        Coordonnées du point d'arrivée.
    """
    P, vec, lum = rayon
    return P + s*vec


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


def coeff_reflection(vecteur_direction_i, vecteur_normal):
    """
    Renvoie le coefficient de réflection du rayon incident vecteur_direction_i réfléchi par la surface de normale vecteur_normal.

    Parameters
    ----------
    vecteur_direction_i : array
        Coordonnées du rayon incident.
    vecteur_normal : array
        Coordonnées du vecteur normal.

    Returns
    -------
    float
        Coefficient de réflection.
    """
    theta_i = np.arccos(-np.dot(vecteur_direction_i, vecteur_normal))
    theta_r = np.arcsin(n1/n2*np.sin(theta_i))
    return 1/2*((np.sin(theta_r-theta_i)**2)/(np.sin(theta_i+theta_r)**2) + (np.tan(theta_r-theta_i)**2)/(np.tan(theta_i+theta_r)**2))



###

def vecteur_normal_ij(surface, i, j):
    A = np.array([i*dx, j*dy, surface[i, j]])
    B = np.array([(i+1)*dx, j*dy, surface[i+1, j]])
    AB = vec(A, B)
    C = np.array([i*dx, (j+1)*dy, surface[i, j+1]])
    AC = vec(A, C)

    vecteur_normal = np.cross(AB, AC)
    vecteur_normal = 1/np.linalg.norm(vecteur_normal)*vecteur_normal
    return vecteur_normal

def vecteurs_normaux_surface(surface):
    """
    Renvoie les vecteurs normaux à la surface en calculant des produits vectoriels.

    Parameters
    ----------
    surface : array
        Hauteur de la surface aux points (i,j).

    Returns
    -------
    array
        Tableau des vecteurs normaux à la surface aux points (i,j).
    """
    vecteurs_normaux = []
    for i in range(Nx-1):
        vecteurs_normaux.append([])
        for j in range(Ny-1):
            vecteur_normal = vecteur_normal_ij(surface, i, j)
            vecteurs_normaux[i].append(vecteur_normal)
    return np.array(vecteurs_normaux)


def test_intersection(rayon, surface, s, vecteurs_normaux, depart):
    """
    Renvoie 0 si le rayon à la distance s appartient à l'interface.

    Parameters
    ----------
    rayon : array
        Objet rayon lumineux.
    surface : array
        Hauteur de la surface aux points (i,j).
    s : float
        Distance.
    vecteurs_normaux : array
        Tableau des vecteurs normaux à la surface aux points (i,j).
    interface : str
        Nom de l'interface.

    Returns
    -------
    float
        Nombre égal à 0 si et seulement si s*vec appartient à l'interface.
    """

    point_interface = rayon.point_rayon(depart, s)

    if depart == "surface":
        P = np.zeros(3)
        vecteur_normal = np.array([0, 0, 1])

    elif depart == "source":
        i, j = indices_du_point(point_interface)
        P = np.array([i*dx, j*dx, surface[i, j]])
        vecteur_normal = vecteurs_normaux[i, j]

    return np.dot(vecteur_normal, vec(P, point_interface))


def find_point_intersection(rayon, surface, vecteurs_normaux, depart):
    """
    Renvoie le point d'intersection du rayon avec l'interface.
    On fait une recherche de zéro de la fonction test_intersection (en fonction de s).

    Parameters
    ----------
    rayon : array
        Objet rayon lumineux.
    surface : array
        Hauteur de la surface aux points (i,j).
    vecteurs_normaux : array
        Tableau des vecteurs normaux à la surface aux points (i,j).
    interface : str
        Nom de l'interface.

    Returns
    -------
    array
        Coordonnées du point d'intersection.
    """
    if depart == "source":
        recherche_zero = scipy.optimize.root_scalar(lambda s: test_intersection(
            rayon, surface, s, vecteurs_normaux, depart), x0=0, x1=Lz)
        s_intersection = recherche_zero.root

    else:
        s_intersection = -rayon.point_interface[2]/rayon.vecteur_direction_r[2]

    point_intersection = rayon.point_rayon(depart, s_intersection)

    return point_intersection




###




class Rayon():
    def __init__(self, point_source, vecteur_direction_i, lum):
        """
        Génère un objet rayon lumineux.

        Parameters
        ----------
        point_source : array
            Point d'où part le rayon.
        vecteur_direction_i : array
            Direction du rayon incident.
        lum : float
            Intensité lumineuse.
        """
        self.point_source = point_source
        self.vecteur_direction_i = vecteur_direction_i
        self.lum = lum

        self.point_interface = None
        self.vecteur_direction_r = None
        self.lum_r = None
        
        self.point_sol = None
        self.lum_sol = None

    def point_rayon(self, depart, s):
        if depart=="source":
            return self.point_source + s*self.vecteur_direction_i
        elif depart=="surface":
            return self.point_interface + s*self.vecteur_direction_r
    
    def refract(self, vecteur_normal):
        self.vecteur_direction_r = refract(self.vecteur_direction_i, vecteur_normal)
        self.lum_r = self.lum * (1 - coeff_reflection(self.vecteur_direction_i, vecteur_normal))
    
    def find_point_intersection(self, rayon, surface, vecteurs_normaux, depart):

        point_intersection = find_point_intersection(rayon, surface, vecteurs_normaux, depart)

        if depart == "surface":
            self.point_sol = point_intersection
            self.lum_sol = self.lum_r * np.exp(-c*np.linalg.norm(vec(self.point_interface, self.point_sol)))
        elif depart == "source":
            self.point_interface = point_intersection
    

