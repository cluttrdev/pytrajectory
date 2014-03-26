# coding: utf8
from __future__ import division
import numpy as np
from numpy.linalg import norm

import sympy as sp

from ipHelp import IPS, ST, ip_syshook, dirsearch, sys

from algopy import UTPM
import algopy


class solver:
    #this class provides numerical methods for solving equation systems
    #needs algopy https://github.com/b45ch1/algopy
    def __init__(self,GLS,x,x0,tol=10e-5,maxx=10,algo='leven'):
        """
        #GLS should be a numpy array of sympy expressions
        #x should be a numpy array of all symbolic variables been used in GLS
        #x0 should be a numpy array of a guess 
        #algo ... newton, gauss, leven
        #tol ... is tolerance for the solver, algorithm stops if norm becomes smaller
        #maxx ... or maximum number of iterations is reached
        """

        self.x0 = np.array(x0,dtype=float)
        n=len(GLS)
        if (not n==len(x0) and algo=='newton'):
            print 'Newton needs square equation systems'
            return x0.tolist()
        
        self.sol=None
        self.x = x
        self.tol = tol
        self.maxx = maxx
        self.algo = algo
        self.n=n
        #IPS()

        #create a callable function for GLS
        self.GLSfoo = sp.lambdify(x,sp.Matrix(GLS),modules='numpy')

        #create callable function for jacobian with automatic differentiation reverse mode
        cg = algopy.CGraph()
        x1 = algopy.Function(self.x0)# dtype von algopy, wert bleibt gleich
        y = self.F(x1) ##*1
        cg.trace_off()
        cg.independentFunctionList = [x1]
        cg.dependentFunctionList = [y]

        self.J=cg.jacobian #this is a callable function

    def solve(self):

        if (self.algo=='newton'):
            print '###############################'
            print '### run newton solver with   ###'
            print '### Automatic Differentiation**###'
            print '###############################'
            self.newton()

        elif (self.algo=='gauss'):
            print '###############################'
            print '### run gauss solver with   ###'
            print '### Automatic Differentiation ###'
            print '###############################'
            self.gauss()

        elif (self.algo=='leven'):
            print '###################################'
            print '### Levenberg-Marquet-Verfahren ###'
            print '###  Automatic Differentiation **###'
            print '###################################'
            self.leven()

        if (self.sol==None):
            print 'Wrong solver'
            return self.x0
        else:
            return self.sol

    def F(self,args): #Arugmente auspacken und func "G" aufrufen
        #a little wrapper due the behavior of lambdify
    
        ##?? Was passiert hier und warum
        # siehe ##*1
    
        out = algopy.zeros(self.n, dtype=args)
        G = self.GLSfoo(*args)

        for i in xrange(self.n):
            out[i]=G[0,i]
        return out


    def newton(self): #func, jac, start

        res=1
        i=0
        x0 = self.x0
        f1 = np.matrix(self.F(x0))
        while(res>self.tol and self.maxx>i):

            i=i+1
            j1 = self.J(x0)

            h=np.array(np.linalg.solve(j1,f1.T).flatten())[0] 

            x0-=h

            #IPS.nest(loc=locals(), glob=globals())
            f1 = np.matrix(self.F(x0))
            res = np.linalg.norm(f1)
            print i,': ',res
        self.sol = x0


    def gauss(self):

        i=0
        x = self.x0
        res=1
        res_alt=10e10
        while((res>self.tol) and (self.maxx>i) and (abs(res-res_alt)>self.tol)):

            i+=1
            r=np.matrix(self.F(x))

            D = np.matrix(self.J(x))
            DD=np.linalg.solve(D.T*D,D.T*r.T) 

            x = np.matrix(x).T - DD
            x= np.array(x.flatten())[0]
            res_alt=res
            res = norm(r)
            print i,': ',res

        self.sol = x


    def leven(self): #func, jac, start

        i=0
        x = self.x0
        res=1
        res_alt=10e10
        
        ny=0.1 ##-> mu
        
        # borders for convergence-control ##!! Ref zu Doku
        b0 = 0.2
        b1 = 0.8
        
        roh  = 0.0

        n = len(self.x0)
        ##?? warum Bed. 1 und 3? (-» retol und abstol)
        while((res>self.tol) and (self.maxx>i) and (abs(res-res_alt)>self.tol)):

            i+=1

            while (roh<b0):
                ## -> np.dot statt matrix-dtype nutzen
                F=np.matrix(self.F(x)).T
                J = np.matrix(self.J(x))
                normFx = norm(F)


                ##?? warum J.T*F? (4.18: J*F)
                ## -» .T gehört eigentlich oben hin
                s=-np.linalg.solve(J.T*J+ny**2*np.eye(n),J.T*F)


                Fxs = self.F(x+np.array(s).flatten())

                #IPS.nest(loc=locals(), glob=globals())

                roh = (normFx**2 - norm(Fxs)**2) / (normFx**2 - (norm(F+J*s))**2)
                print 'Roh:', roh
                if (roh<=b0): ny = 1.5*ny
                if (roh>=b1): ny = 0.75*ny
                print 'ny:', ny
            #IPS()
            roh = 0.0
            x = x + np.array(s).flatten()
            #IPS()
            #x= np.array(x.flatten())[0]
            res_alt=res
            res = normFx
            print i,': ',res

        self.sol = x

if __name__ == '__main__':

    from ipHelp import IPS, ST, ip_syshook, dirsearch, sys
    from math import sin,cos,exp
    from time import clock

    x1,x2,x3  =sp.symbols('x1 x2 x3')

    x=np.array([x1,x2,x3])

    GLS = [ x2+x3-5,
            -x1*2 +x3 +1, 
            -x2 + x1 -x3 + 7]
            #x1**2 +x2**2]

    #x0=list([-1.1,9,-3])
    x0=np.array([100.0,1.0,1.0])
    #IPS()
    t1=clock()
    GLS = np.array(GLS)
#    solver = solver(GLS,x,x0,algo='gauss',tol=1e-10)
    solver = solver(GLS,x,x0,algo='leven',tol=1e-10)
    sol = solver.solve()
    print clock()-t1
    print sol
    IPS()