# coding: utf8
import numpy as np
import sympy as sp
from ipHelp import IPS, ST, ip_syshook, dirsearch, sys


def clean(GLS,tol=10e-10):
    """
    ???
    """

    ll=0
    para=0;
    loop=1

    while(loop): #es werde nicht alle atoms beim ersten mal erwischt?
        loop=0
        ll=ll+1
        if (ll>100):
            print 'Loop gut stuck while filter terms<10e-8'
            break

        if (type(GLS)!=list):
            atoms=list(GLS.atoms())
            for ato in atoms:
                if (abs(ato)<tol):
                    loop=0 #still something to do
                    para=para+1
                    GLS = GLS.subs(ato,0.0)

        else:
            for i,G in enumerate(GLS):
                atoms=list(G.atoms())
                for ato in atoms:
                    if (abs(ato)<tol):
                        loop=0 #still something to do
                        para=para+1
                        G = G.subs(ato,0.0)
                        GLS[i]=G

    return GLS


def Sym2NumArray(F):
    """
    convert sympy Matrix to np array
    """
    
    ##!! geht bestimmt schneller und einfacher

    shapeF=sp.shape(F)
    B=np.zeros(shapeF)
    for i in range(0,sp.shapeF[0]):
        for j in range(0,shapeF[1]):
            B[i,j]=sp.N(F[i,j])
    return B

class BetweenDict(dict):
    ##?? Quelle? Lizenz?
    def __init__(self, d = {}):
        for k,v in d.items():
            self[k] = v

    def __getitem__(self, key):

        if (key<=0.0):
            key=10e-10 #sehr unschön

        for k, v in self.items():
            if k[0] < key <= k[1]:
                return v
        raise KeyError("Key '%s' is not between any values in the BetweenDict" % key)

    def __setitem__(self, key, value):
        try:
            if len(key) == 2:
                if key[0] < key[1]:
                    dict.__setitem__(self, (key[0], key[1]), value)
                else:
                    raise RuntimeError('First element of a BetweenDict key '
                                       'must be strictly less than the '
                                       'second element')
            else:
                raise ValueError('Key of a BetweenDict must be an iterable '
                                 'with length two')
        except TypeError:
            raise TypeError('Key of a BetweenDict must be an iterable '
                             'with length two')

    def __contains__(self, key):
        try:
            return bool(self[key]) or True
        except KeyError:
            return False



def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3


    ##?? was bedeuten die Zahlen bei dash[...]?
    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)