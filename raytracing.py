
## Imports

import numpy as np

from constantes import *


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
    return 2*projection(v, n) - v


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
    """
    Calcul de l'angle de réfraction en fonction de l'angle d'incidence.

    Parameters
    ----------
    cos_theta1 : float
        Angle d'incidence

    Returns
    -------
    float
        Angle de réfraction
    """
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
    rr = n1/n2*ri + (n1/n2*cos_theta1 - cos_theta_refract(cos_theta1))*n
    rr = 1/np.linalg.norm(rr)*rr
    return rr


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


def coeff_reflection(ri, n):
    """
    Renvoie le coefficient de réflection du rayon incident ri réfléchi par la surface de normale n.

    Parameters
    ----------
    ri : Array numpy
        Coordonnées du rayon incident
    n : Array numpy
        Coordonnées du vecteur normal

    Returns
    -------
    float
        Coefficient de réflection
    """
    theta_i = np.arccos(-np.dot(ri, n))
    theta_r = np.arcsin(n1/n2*np.sin(theta_i))
    return 1/2*((np.sin(theta_r-theta_i)**2)/(np.sin(theta_i+theta_r)**2) + (np.tan(theta_r-theta_i)**2)/(np.tan(theta_i+theta_r)**2))
