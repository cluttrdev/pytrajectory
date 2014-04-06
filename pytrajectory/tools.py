import numpy as np
import sympy as sp
import scipy as scp

import pylab as plt
from sympy.core.symbol import Symbol
from IPython import embed as IPS

from numpy import sin,cos
import matplotlib as mpl


class IntegChain():
    '''
    This class provides a representation of a integrator chain consisting of sympy symbols ...
    '''
    
    def __init__(self, lst):
        self.elements = lst
        self.upper = self.elements[0]
        self.lower = self.elements[-1]
    
    def __len__(self):
        return len(self.elements)
    
    def __getitem__(self, key):
        return self.elements[key]
    
    def __contains__(self, item):
        return (item in self.elements)
    
    def __str__(self):
        s = ''
        for elem in self.elements[::-1]:
            s += ' -> ' + elem.name
        return s[4:]
    
    def successor(self, elem):
        '''
        This method returns the successor of the given element of the
        integrator chains, i.e. it returns :math:`\\frac{d}{dt}[elem]`
        
        :param sympy.Symbol elem: An element of the integrator chain
        '''
        if not elem == self.bottom:
            i = self.elements.index(elem)
            succ = self.elements[i+1]
        else:
            print 'ERROR: lower end of integrator chain has no successor'
            succ = elem
        
        return succ
    
    def predecessor(self, elem):
        '''
        This method returns the predecessor of the given element of the
        integrator chains, i.e. it returns :math:`\\int [elem]`
        
        :param sympy.Symbol elem: An element of the integrator chain
        '''
        if not elem == self.top:
            i = self.elements.index(elem)
            pred = self.elements[i-1]
        else:
            print 'ERROR: uper end of integrator chain has no successor'
            pred = elem
        
        return pred


class Grid():
    def __init__(self):
        pass


class BetweenDict(dict):
    ##?? Quelle? Lizenz?
    def __init__(self, d = {}):
        for k,v in d.items():
            self[k] = v

    def __getitem__(self, key):

        if (key<=0.0):
            key=10e-10 #sehr unschoen

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


class struct():
    def __init__(self):
        return


class Modell:
    def __init__(self):
        self.fig=plt.figure()
        self.ax=plt.axes()
    
        mng = plt.get_current_fig_manager()
     
        mng.window.setGeometry(0, 0, 1000, 700)
    
        self.ax.set_xlim(-1.2,0.3)
        self.ax.set_ylim(-0.6,0.6)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        self.ax.set_position([0.01,0.01,0.98,0.98])
        self.ax.set_frame_on(True)
        self.ax.set_aspect('equal')
        self.ax.set_axis_bgcolor('w')
    
        self.image=0

    def draw(self,x,phi,frame,image=0):
        L=0.5
        
        car_width=0.05
        car_heigth = 0.02
        pendel_size = 0.015
        
        x_car=x
        y_car=0
        
        x_pendel=-L*sin(phi)+x_car
        y_pendel= L*cos(phi)
        
        if (image==0):
            # init
            image=struct()
        else:
            # update
            image.sphere.remove()
            image.stab.remove()
            image.car.remove()
            image.gelenk.remove()
        
        #Ball
        image.sphere=mpl.patches.Circle((x_pendel,y_pendel),pendel_size,color='k')
        self.ax.add_patch(image.sphere)
        
        #Car
        image.car=mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
        self.ax.add_patch(image.car)
        #IPS()
        image.gelenk=mpl.patches.Circle((x_car,0),0.005,color='k')
        self.ax.add_patch(image.gelenk)
        #self.ax.annotate(frame, xy=(x_pendel, y_pendel), xytext=(x_pendel+0.02, y_pendel))
        #Stab
        image.stab=self.ax.add_line(mpl.lines.Line2D([x_car,x_pendel],[y_car,y_pendel],color='k',zorder=1,linewidth=2.0))
        
        #txt = plt.text(x_pendel+0.05,y_pendel,frame)
        
        self.image = image
        
        plt.draw()


def blockdiag(M, bshape=None, sparse=False):
    '''
    Takes block of shape :attr:`shape`  from matrix :attr:`M` and creates 
    block diagonal matrix out of them.
    
    Parameters
    ----------
    
    M : numpy.ndarray
        Matrix to take blocks from
    bshape : tuple
        Shape of one block
    sparse : bool
        Whether or not to return created matrix as sparse matrix
    
    Examples
    --------
    
    >>> A = np.ones((4, 2))
    >>> print A
    [[ 1.  1.]
     [ 1.  1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> B = blockdiag(A, (2, 2))
    >>> print B
    [[ 1.  1.  0.  0.]
     [ 1.  1.  0.  0.]
     [ 0.  0.  1.  1.]
     [ 0.  0.  1.  1.]]
    '''
    
    if type(M) == 'list':
        pass
    else:
        nrow, ncol = bshape
        Mrow, Mcol = M.shape
        
        if not Mcol == ncol:
            print 'ERROR: ncol /= #col of M'
            return M
        if not Mrow % nrow == 0:
            print 'ERROR: nrow has to be teiler of #row of M'
            return M
        
        n = Mrow / nrow
        Mb = np.zeros((Mrow, n*ncol))
        
        for i in xrange(n):
            Mb[i*nrow : (i+1)*nrow, i*ncol : (i+1)*ncol] = M[i*nrow : (i+1)*nrow, :]
    
    if sparse:
        Mb = scp.sparse.csr_matrix(Mb)
    
    return Mb



def plot(self):
    #provides graphics for each system variable, manipulated variable and error function
    #plots the solution of the simulation
    #at the end the error at the final state will be calculated
    
    log.info("Plot")
    
    z=self.n+self.m+len(self.eqind)
    z1=np.floor(np.sqrt(z))
    z2=np.ceil(z/z1)
    t=self.A[0]
    xt = self.A[1]
    ut= self.A[2]
    
    
    log.info("Ending up with:")
    for i,xx in enumerate(self.x_sym):
        log.info(str(xx)+" : "+str(xt[-1:][0][i]))
    
    log.info("Shoul be:")
    for i,xx in enumerate(self.x_sym):
        log.info(str(xx)+" : "+str(self.xb[xx]))
    
    log.info("Difference")
    for i,xx in enumerate(self.x_sym):
        log.info(str(xx)+" : "+str(self.xb[xx]-xt[-1:][0][i]))
    
    
    if not __name__=='__main__':
        return
    
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
    
    
    plt.rcParams['figure.subplot.bottom']=.2
    plt.rcParams['figure.subplot.top']= .95
    plt.rcParams['figure.subplot.left']=.13
    plt.rcParams['figure.subplot.right']=.95
    
    plt.rcParams['font.size']=16
    
    plt.rcParams['legend.fontsize']=16
    plt.rc('text', usetex=True)
    
    
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.rcParams['legend.fontsize']=20
    
    plt.rcParams['axes.titlesize']=26
    plt.rcParams['axes.labelsize']=26
    
    
    plt.rcParams['xtick.major.pad']='8'
    plt.rcParams['ytick.major.pad']='8'
    
    mm = 1./25.4 #mm to inch
    scale = 3
    fs = [100*mm*scale, 60*mm*scale]
    
    fff=plt.figure(figsize=fs, dpi=80)
    
    
    PP=1
    for i,xx in enumerate(self.x_sym):
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,xt[:,i])
        plt.xlabel(r'$t$')
        plt.title(r'$'+str(xx)+'(t)$')
    
    for i,uu in enumerate(self.u_sym):
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,ut[:,i])
        plt.xlabel(r'$t$')
        plt.title(r'$'+str(uu)+'(t)$')
    
    for hh in self.H:
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,self.H[hh])
        plt.xlabel(r'$t$')
        plt.title(r'$H_'+str(hh+1)+'(t)$')
    
    setFigLinesBW(fff)
    
    plt.tight_layout()
    
    plt.show()
