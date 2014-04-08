import numpy as np
import sympy as sp
import scipy as scp

from sympy.core.symbol import Symbol
from IPython import embed as IPS

from numpy import sin,cos
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation


class IntegChain():
    '''
    This class provides a representation of a integrator chain consisting of sympy symbols ...
    
    
    Parameters
    ----------
    
    lst : lst
        Ordered list of elements for the integrator chain
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
        
        Parameters
        ----------
        elem : sympy.Symbol
            An element of the integrator chain
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
        
        
        Parameters
        ----------
        elem : sympy.Symbol
            An element of the integrator chain
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


class Animation():
    '''
    Provides animation capabilities.
    
    Parameters
    ----------
    
    drawfnc : callable
        Function that returns an image of the current system state according to :attr:`data`
    simdata : numpy.ndarray
        Array that contains simulation data (time, system states, input states)
    '''
    
    def __init__(self, drawfnc, simdata):
        self.fig = plt.figure()
        self.ax = plt.axes()
    
        #mng = plt.get_current_fig_manager()
     
        #mng.window.setGeometry(0, 0, 1000, 700)
    
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_frame_on(True)
        self.ax.set_aspect('equal')
        self.ax.set_axis_bgcolor('w')
    
        self.image = 0
        
        self.nframes = 80
        
        self.draw = drawfnc
        self.data = simdata
    
    
    class Image():
        def __init__(self):
            self.patches = []
            self.lines = []
        
        def reset(self):
            self.patches = []
            self.lines = []
    
    
    def set_limits(self, xlim, ylim):
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
    
    
    def set_pos(self, pos):
        self.ax.set_position(pos)
    
    
    def animate(self):
        t = self.data[0]
        xt = self.data[1]
        
        tt = np.linspace(0,(len(t)-1),self.nframes+1,endpoint=True)
        
        self.T = t[-1] - t[0]
        
        def _animate(frame):
            i = tt[frame]
            print frame
            image = self.image
            
            if image == 0:
                # init
                image = self.Image()
            else:
                # update
                for p in image.patches:
                    p.remove()
                for l in image.lines:
                    l.remove()
                image.reset()
            
            self.image = self.draw(xt[i,:], image=image)
            
            for p in self.image.patches:
                self.ax.add_patch(p)
            
            for l in self.image.lines:
                self.ax.add_line(l)
            
            plt.draw()
            
        
        self.anim = animation.FuncAnimation(self.fig, _animate, frames=self.nframes,
                                                interval=1, blit=True)
    
    
    def save(self, fname, fps=None):
        if not fps:
            fps = self.nframes/float(self.T)
        self.anim.save(fname, fps=fps)


def blockdiag(M, bshape=None, sparse=False):
    '''
    Takes block of shape :attr:`bshape`  from matrix :attr:`M` and creates 
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
            print 'ERROR: nrow has to be a factor of #row of M'
            return M
        
        n = Mrow / nrow
        Mb = np.zeros((Mrow, n*ncol))
        
        for i in xrange(n):
            Mb[i*nrow : (i+1)*nrow, i*ncol : (i+1)*ncol] = M[i*nrow : (i+1)*nrow, :]
    
    if sparse:
        Mb = scp.sparse.csr_matrix(Mb)
    
    return Mb



def plot(sim, H, fname=None):
    '''
    This method provides graphics for each system variable, manipulated
    variable and error function and plots the solution of the simulation.
    
    
    Parameters
    ----------
    
    sim : tuple
        Contains collocation points, and simulation results of system and input variables
    H : dict
        Dictionary of the callable error functions
    fname : str
        If not None, plot will be saved as <fname>.png
    '''
    
    t, xt, ut = sim
    n = xt.shape[1]
    m = ut.shape[1]
    
    z = n + m + len(H.keys())
    z1 = np.floor(np.sqrt(z))
    z2 = np.ceil(z/z1)

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
    for i in xrange(n):
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,xt[:,i])
        plt.xlabel(r'$t$')
        plt.title(r'$'+'x%d'%i+'(t)$')

    for i in xrange(m):
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,ut[:,i])
        plt.xlabel(r'$t$')
        plt.title(r'$'+'u%d'%i+'(t)$')

    for hh in H:
        plt.subplot(int(z1),int(z2),PP)
        PP+=1
        plt.plot(t,H[hh])
        plt.xlabel(r'$t$')
        plt.title(r'$H_'+str(hh+1)+'(t)$')

    plt.tight_layout()

    plt.show()
    
    if fname:
        if not fname.endswith('.png'):
            plt.savefig(fname+'.png')
        else:
            plt.savefig(fname)


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
