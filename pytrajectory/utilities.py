import numpy as np
import sympy as sp
import scipy as scp

from sympy.core.symbol import Symbol
from IPython import embed as IPS

from numpy import sin,cos
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec


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


class Animation():
    '''
    Provides animation capabilities.
    
    
    Parameters
    ----------
    
    drawfnc : callable
        Function that returns an image of the current system state according to :attr:`data`
    simdata : numpy.ndarray
        Array that contains simulation data (time, system states, input states)
    plotsys : list
        List of tuples with indices and labels of system variables that will be plotted along the picture
    plotinputs : list
        List of tuples with indices and labels of input variables that will be plotted along the picture
    '''
    
    def __init__(self, drawfnc, simdata, plotsys=[], plotinputs=[]):
        self.fig = plt.figure()
    
        self.image = 0
        
        self.t = simdata[0]
        self.xt = simdata[1]
        self.ut = simdata[0]
        
        self.plotsys = plotsys
        self.plotinputs = plotinputs
        
        self.get_axes()
        
        #if plotcurves:
        #    self.plot_curves = True
        #else:
        #    self.plot_curves = False
        
        self.axes['ax1'].set_xticks([])
        self.axes['ax1'].set_yticks([])
        self.axes['ax1'].set_frame_on(True)
        self.axes['ax1'].set_aspect('equal')
        self.axes['ax1'].set_axis_bgcolor('w')
        
        self.nframes = 80
        
        self.draw = drawfnc
    
    
    class Image():
        def __init__(self):
            self.patches = []
            self.lines = []
        
        def reset(self):
            self.patches = []
            self.lines = []
    
    
    def get_axes(self):
        sys = self.plotsys
        inputs = self.plotinputs
        
        if not sys+inputs:
            gs = GridSpec(1,1)
        else:
            #gs = GridSpec(len(sys+inputs), 2)
            gs = GridSpec(2, len(sys+inputs))
        
        axes = dict()
        syscurves = []
        inputcurves = []
        
        #axes['ax1'] = self.fig.add_subplot(gs[:,0])
        axes['ax1'] = self.fig.add_subplot(gs[0,:])
        
        for i in xrange(len(sys)):
            #axes['ax%d'%(i+2)] = self.fig.add_subplot(gs[i,1])
            axes['ax%d'%(i+2)] = self.fig.add_subplot(gs[1,i])
            
            curve = mpl.lines.Line2D([], [], color='black')
            syscurves.append(curve)
            
            axes['ax%d'%(i+2)].add_line(curve)
        
        offset = len(sys)
        for i in xrange(len(inputs)):
            #axes['ax%d'%(i+2+offset)] = self.fig.add_subplot(gs[i,1])
            axes['ax%d'%(i+2+offset)] = self.fig.add_subplot(gs[1,i+offset])
            
            curve = mpl.lines.Line2D([], [], color='black')
            inputcurves.append(curve)
            
            axes['ax%d'%(i+2+offset)].add_line(curve)
        
        self.axes = axes
        self.syscurves = syscurves
        self.inputcurves = inputcurves
    
    
    def set_limits(self, ax='ax1', xlim=(0,1), ylim=(0,1)):
        self.axes[ax].set_xlim(*xlim)
        self.axes[ax].set_ylim(*ylim)
    
    
    def set_label(self, ax='ax1', label=''):
        #self.axes[ax].set_xlabel(xlabel)
        self.axes[ax].set_title(label)
        
    
    def animate(self):
        t = self.t
        xt = self.xt
        ut = self.ut
        
        tt = np.linspace(0,(len(t)-1),self.nframes+1,endpoint=True)
        
        self.T = t[-1] - t[0]
        
        # set axis limits and labels of system curves
        xlim = (0.0, self.T)
        for i, idxlabel in enumerate(self.plotsys):
            idx, label = idxlabel
            
            try:
                ylim = (min(xt[:,idx]), max(xt[:,idx]))
            except:
                ylim = (min(xt), max(xt))
            
            self.set_limits(ax='ax%d'%(i+2), xlim=xlim, ylim=ylim)
            self.set_label(ax='ax%d'%(i+2), label=label)
            
        # set axis limits and labels of input curves
        offset = len(self.plotsys)
        for i, idxlabel in enumerate(self.plotinputs):
            idx, label = idxlabel
            
            try:
                ylim = (min(ut[:,idx]), max(ut[:,idx]))
            except:
                ylim = (min(ut), max(ut))
            
            self.set_limits(ax='ax%d'%(i+2), xlim=xlim, ylim=ylim)
            self.set_label(ax='ax%d'%(i+2), label=label)
        
        def _animate(frame):
            i = tt[frame]
            print frame
            
            # draw picture
            image = self.image
            ax1 = self.axes['ax1']
            
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
            
            image = self.draw(xt[i,:], image=image)
            
            for p in image.patches:
                ax1.add_patch(p)
            
            for l in image.lines:
                ax1.add_line(l)
            
            self.image = image
            self.axes['ax1'] = ax1
            
            # draw system curves
            for k, curve in enumerate(self.syscurves):
                try:
                    curve.set_data(t[:i], xt[:i,self.plotsys[k][0]])
                except:
                    curve.set_data(t[:i], xt[:i])
                self.axes['ax%d'%(k+2)].add_line(curve)
            
            # draw input curves
            for k, curve in enumerate(self.inputcurves):
                try:
                    curve.set_data(t[:i], ut[:i,self.plotinputs[k][0]])
                except:
                    curve.set_data(t[:i], ut[:i])
                self.axes['ax%d'%(k+2+offset)].add_line(curve)
            
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
