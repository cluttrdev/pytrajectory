# IMPORTS
import numpy as np

from splines import CubicSpline
import log




class Trajectory(object):
    '''
    here comes the docstring...
    '''
    
    def __init__(self, CtrlSys):
        self.sys = CtrlSys
        self.indep_coeffs = []
        self.splines = dict()
        self.x_fnc = dict()
        self.u_fnc = dict()
        self.dx_fnc = dict()
    
    
    def x(self, t):
        '''
        Returns the current system state.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the system at.
        '''
        
        if not self.sys.a <= t <= self.sys.b:
            log.warn("Time point 't' has to be in (a,b)", verb=3)
            arr = None
        else:
            arr = np.array([self.x_fnc[xx](t) for xx in self.sys.x_sym])
        
        return arr


    def u(self, t):
        '''
        Returns the state of the input variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the input variables at.
        '''
        
        if not self.sys.a <= t <= self.sys.b+0.05:
            log.warn("Time point 't' has to be in (a,b)", verb=3)
            arr = None
            arr = np.array([self.u_fnc[uu](self.sys.b) for uu in self.sys.u_sym])
        else:
            arr = np.array([self.u_fnc[uu](t) for uu in self.sys.u_sym])
        
        return arr


    def dx(self, t):
        '''
        Returns the state of the 1st derivatives of the system variables.
        
        Parameters
        ----------
        
        t : float
            The time point in (a,b) to evaluate the 1st derivatives at.
        '''
        
        if not self.sys.a <= t <= self.sys.b+0.05:
            log.warn("Time point 't' has to be in (a,b)", verb=3)
            arr = None
        else:
            arr = np.array([self.dx_fnc[xx](t) for xx in self.sys.x_sym])
        
        return arr
    
    
    
    def initSplines(self):
        '''
        here comes the docstring...
        '''
        
        log.info("  Initialise Splines", verb=2)
        
        # dictionaries for splines and callable solution function for x, u and dx
        splines = dict()
        x_fnc = dict()
        u_fnc = dict()
        dx_fnc = dict()
        
        # make some stuff local
        sx = self.sys.mparam['sx']
        su = self.sys.mparam['su']

        # first handle variables that are part of an integrator chain
        for chain in self.sys.chains:
            upper = chain.upper
            lower = chain.lower

            # here we just create a spline object for the upper ends of every chain
            # w.r.t. its lower end
            if lower.name.startswith('x'):
                splines[upper] = CubicSpline(self.sys.a, self.sys.b, n=sx,
                                             bc=[self.sys.xa[upper], self.sys.xb[upper]],
                                             steady=False, tag=upper.name)
                splines[upper].type = 'x'
            elif lower.name.startswith('u'):
                splines[upper] = CubicSpline(self.sys.a, self.sys.b, n=su,
                                             bc=[self.sys.ua[lower], self.sys.ub[lower]],
                                             steady=False, tag=upper.name)
                splines[upper].type = 'u'

            for i, elem in enumerate(chain.elements):
                if elem in self.sys.u_sym:
                    if (i == 0):
                        u_fnc[elem] = splines[upper].f
                    if (i == 1):
                        u_fnc[elem] = splines[upper].df
                    if (i == 2):
                        u_fnc[elem] = splines[upper].ddf
                elif elem in self.sys.x_sym:
                    if (i == 0):
                        splines[upper].bc = [self.sys.xa[elem], self.sys.xb[elem]]
                        if splines[upper].type == 'u':
                            splines[upper].bcd = [self.sys.ua[lower], self.sys.ub[lower]]
                        x_fnc[elem] = splines[upper].f
                    if (i == 1):
                        splines[upper].bcd = [self.sys.xa[elem], self.sys.xb[elem]]
                        if splines[upper].type == 'u':
                            splines[upper].bcdd = [self.sys.ua[lower], self.sys.ub[lower]]
                        x_fnc[elem] = splines[upper].df
                    if (i == 2):
                        splines[upper].bcdd = [self.sys.xa[elem], self.sys.xb[elem]]
                        x_fnc[elem] = splines[upper].ddf

        # now handle the variables which are not part of any chain
        for xx in self.sys.x_sym:
            if (not x_fnc.has_key(xx)):
                splines[xx] = CubicSpline(self.sys.a, self.sys.b, n=sx,
                                          bc=[self.sys.xa[xx], self.sys.xb[xx]],
                                          steady=False, tag=str(xx))
                splines[xx].type = 'x'
                x_fnc[xx] = splines[xx].f

        for i, uu in enumerate(self.sys.u_sym):
            if (not u_fnc.has_key(uu)):
                splines[uu] = CubicSpline(self.sys.a, self.sys.b, n=su,
                                          bc=[self.sys.ua[uu], self.sys.ub[uu]],
                                          steady=False, tag=str(uu))
                splines[uu].type = 'u'
                u_fnc[uu] = splines[uu].f
        
        # solve smoothness conditions of each spline
        for ss in splines:
            with log.Timer("makesteady()"):
                splines[ss].makesteady()

        for xx in self.sys.x_sym:
            dx_fnc[xx] = fdiff(x_fnc[xx])

        indep_coeffs= dict()
        for ss in splines:
            indep_coeffs[ss] = splines[ss].c_indep

        self.indep_coeffs = indep_coeffs
        self.splines = splines
        self.x_fnc = x_fnc
        self.u_fnc = u_fnc
        self.dx_fnc = dx_fnc
    
    
    def setCoeff(self):
        '''
        Set found numerical values for the independent parameters of each spline.

        This method is used to get the actual splines by using the numerical
        solutions to set up the coefficients of the polynomial spline parts of
        every created spline.
        '''

        log.info("    Set spline coefficients", verb=2)

        sol = self.sys.sol
        subs = dict()

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            i = len(v)
            subs[k] = sol[:i]
            sol = sol[i:]

        for var in self.sys.x_sym + self.sys.u_sym:
            for ic in self.sys.chains:
                if var in ic:
                    subs[var] = subs[ic.upper]

        # set numerical coefficients for each spline
        for cc in self.splines:
            self.splines[cc].set_coeffs(subs[cc])

        # yet another dictionary for solution and coeffs
        coeffs_sol = dict()

        # used for indexing
        i = 0
        j = 0

        for k, v in sorted(self.indep_coeffs.items(), key=lambda (k, v): k.name):
            j += len(v)
            coeffs_sol[k] = self.sys.sol[i:j]
            i = j

        self.coeffs_sol = coeffs_sol
    
    
    def checkAccuracy(self):
        '''
        Checks whether the desired accuracy for the boundary values was reached.

        It calculates the difference between the solution of the simulation
        and the given boundary values at the right border and compares its
        maximum against the tolerance set by self.eps
        
        Returns
        -------
        
        bool
            Whether the desired tolerance is satisfied or not.
        '''
        
        # this is the solution of the simulation
        xt = self.sys.sim[1]

        # what is the error
        log.info(40*"-", verb=3)
        log.info("Ending up with:   Should Be:  Difference:", verb=3)

        err = np.empty(self.n)
        if self.sys.constraints:
            for i, xx in enumerate(self.orig_backup['x_sym']):
                err[i] = abs(self.orig_backup['xb'][xx] - xt[-1][i])
                log.info(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.orig_backup['xb'][xx], err[i]), verb=3)
        else:
            for i, xx in enumerate(self.sys.x_sym):
                err[i] = abs(self.sys.xb[xx] - xt[-1][i])
                log.info(str(xx)+" : %f     %f    %f"%(xt[-1][i], self.sys.xb[xx], err[i]), verb=3)
        
        log.info(40*"-", verb=3)
        
        if self.sys.mparam['ierr']:
            # calculate maximum consistency error on the whole interval
            maxH = consistency_error((self.sys.a,self.sys.b), self.x, self.u, self.dx, self.sys.ff)
            
            reached_accuracy = (maxH < self.sys.mparam['ierr']) and (max(err) < self.sys.mparam['eps'])
            log.info('maxH = %f'%maxH)
        else:
            # just check if tolerance for the boundary values is satisfied
            reached_accuracy = max(err) < self.mparam['eps']
        
        if reached_accuracy:
            log.info("  --> reached desired accuracy: "+str(reached_accuracy), verb=1)
        else:
            log.info("  --> reached desired accuracy: "+str(reached_accuracy), verb=2)
        
        return reached_accuracy
    
        