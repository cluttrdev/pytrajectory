# coding: utf8

import numpy as np
import sympy as sp
from ipHelp import IPS, ST, ip_syshook, dirsearch, sys
from tools import BetweenDict ,clean
from time import clock


# import theano as theano ##?? alternative zu algopy, parallel
# import os
# os.environ['SYMPY_USE_CACHE']='no'



def getIndexSymArray(Array, Item):
    ##??
    # 2d array, element -» Index-Paar des ersten auftretens
    # wird in splinklasse an stelle von subs verwendet

    #little function which returns the index of a item in an array
    #is used later for the array, which stores the coefficients of the spline
    for i,AA in enumerate(Array):
        if ((set(AA)).issuperset([Item])):
            j=AA.tolist().index(Item)
            return (i,j)
    print 'Item not in list'
    return None


def diff_func(func,RB=False,traj=None):
    #this function is used to get derivative of of a callable splinefunction (f,d,dd)
    #is used by trajectory
    ##!! im_func ist Funktions-ID
    ## im_self ist das object, von dem func die methode ist

        if(func.im_func == CubicSpline.f.im_func):
            return func.im_self.d
        if(func.im_func == CubicSpline.d.im_func):
            return func.im_self.dd
        if(func.im_func == CubicSpline.dd.im_func):
            return func.im_self.ddd
        
        # raise notImplemented
        print 'not implemented'
        return None


class CubicSpline:
    #provides a object spline
    #C = CubicSpline(0,1,10,tag='x1',RB=[0,0])
    #a Spline in range 0 to 1 with 10 pieces, named 'x1', boundary values: 0 both at 0 and 1
    #call C.makesteady() to solve smoothness and boundary value conditions
    def __init__(self,a,b,n=1,tag='',RB=None,RBd=None,RBdd=None,steady=True):
        #n ... number of spline parts
        #[a,b] ... interval
        #tag ... string of name
        #RB ... Boundary Values for the borders interval
        #RBd ... Boundary Values for the first derivative
        #RBdd ... Boundary Values for the second Derivative
        self.n = int(n)
        self.a = a
        self.b = b

        self.tag = tag
        self.RB=RB
        self.RBd=RBd
        self.RBdd=RBdd
        self.type=None #is being used by trajectory

        print '#######################'
        print '######', tag, '########'
        print '#######################'

        #h is the size of one polynomial part
        self.h = (float(b)-float(a))/float(n)


        self.P=BetweenDict()    #all polynomial parts will be stored in a BetweenDict, 
                                #like normal Dict but accepts values in bewtween interger keys and takes the closest smaller key
        

        self.k=sp.symarray('k'+tag,(int(n),4)) #all spline coefficietns will be stored in a [n x 4]-array
                                                # 4 for each spline part
        
        #fill the Dict
        for i in xrange(int(n)):
            self.P[i,(i+1)]=(np.poly1d(self.k[i,:]),i)

        if (steady):
            self.makesteady()
    

    def diff(self,xi,d):
        #returns the value or derivative value (or symbolic expression) of the spline at a specific point 
        p,i = self.P[xi*self.n/(self.b)]
        return p.deriv(d)(xi-(i+1)*self.h)

    def f(self,xi):
        #just a wrapper für diff 0
        return self.diff(xi,0)

    def d(self,xi):
        #wrapper for diff 1
        return self.diff(xi,1)

    def dd(self,xi):
        #wrapper for diff 2
        return self.diff(xi,2)

    def ddd(self,xi):
        #wrapper for diff 3
        return self.diff(xi,3)

    def setCoeff(self,k_dic):
        #this function is used to set up numerical values for 
        #spline coefficient 
        #just needs a dictionary with coefficient and value
        k_sym = self.k_sym
        n=self.n
        k = self.k

        for i,kk in enumerate(k):
            for j in xrange(4):
                kk[j]=kk[j].evalf(subs=k_dic)
            k[i]=np.array(kk,dtype=float)

        for i in xrange(int(n)):
            self.P[i,(i+1)]=(np.poly1d(k[i]),i)
        self.k=k

    def makesteady(self):

        if (self.n==1): 
            self.k_sym = list(np.array(self.k).flatten())
            return


        print '######################'
        print 'make steady:', self.tag

        h=self.h
        n=int(self.n)
        k=self.k
        Poly=self.P

        
        #mu represents the degree of boundary values
        mu=-1
        if (self.RB!=None): mu+=1
        if (self.RBd!=None): mu+=1
        if (self.RBdd!=None): mu+=1
        
        v = 0 
        if (mu==-1):
            a=k[:,v]
            a=np.hstack((a,k[0,list(set([0,1,2,3])-set([v]))]))
        if (mu==0):
            a=k[:,v]
            a=np.hstack((a,k[0,2]))
        if (mu==1):
            a=k[:-1,v]
        if (mu==2):
            a=k[:-3,v]

        #b is what is not in a
        b=np.array(list(set(k.flatten()) -set(a)) )


        #setting up the equation system for smoothness conditions ans boundary values
        M=[]
        r=[]

        #smoothness for each joining point
        for i in xrange(1,n):
            M.append(Poly[i][0](0)-Poly[i+1][0](-h))
            M.append(Poly[i][0].deriv(1)(0)-Poly[i+1][0].deriv(1)(-h))
            M.append(Poly[i][0].deriv(2)(0)-Poly[i+1][0].deriv(2)(-h))

            r.append(0)
            r.append(0)
            r.append(0)

        #boundary values
        if (self.RB!=None):
            M.append(self.f(self.a))
            M.append(self.f(self.b))
            r.append(self.RB[0])
            r.append(self.RB[1])
        if (self.RBd!=None):
            M.append(self.d(self.a))
            M.append(self.d(self.b))
            r.append(self.RBd[0])
            r.append(self.RBd[1])
        if (self.RBdd!=None):
            M.append(self.dd(self.a))
            M.append(self.dd(self.b))
            r.append(self.RBdd[0])
            r.append(self.RBdd[1])


        #get the A and B matrix
        A=np.matrix(sp.Matrix(M).jacobian(a))
        B=np.matrix(sp.Matrix(M).jacobian(b))
        
        #IPS.nest(loc=locals(), glob=globals())

        #do the inversion
        r=np.matrix(r).T
        T=np.linalg.solve(B,-A)
        T1=np.linalg.solve(B,r)

        #those are the equation of the components in b
        b_subs = T1+T*np.matrix(a).T

        #IPS.nest(loc=locals(), glob=globals())
                

        #take the ralations of b_subs and put them in to our coefficients matrix
        for i,bb in enumerate(b):
            ##!! b: vektor vom LGS wie in Arbeit
            ##self.k n x 4 array mit 4 spline coeffs pro zeile
            ii=getIndexSymArray(self.k,bb)# ii is a tuple
            self.k[ii]=b_subs[i,0]# faster than subs
            #self.k.subs(zip(b, b_subs)) ## oder so ähnlich

        #initialise the polynomials again
        for i in xrange(int(n)):
            self.P[i,(i+1)]=(np.poly1d(self.k[i,:]),i)

        self.k_sym = a

        print '--> done'


if __name__ == '__main__':

    #C1 = CubicSpline(0,1,3)
    t1 = clock()
    C1 = CubicSpline(0,1,100,tag='x1',RB=[0,0],RBd=[0,0],RBdd=[0,0])
    #C1.makesteady()
    #C1 = CubicSpline(0,1,2,tag='x1',RB=[0,0])
    C1.makesteady()
    print clock()-t1
    #C2 = CubicSpline(0,1,5)

    IPS()



##?? ist dieser Code wichtig?



        # for i in range(n):

        #     sub = Poly[i].k
        #     for j in range(4):
        #         sub[j]=sub[j].subs(b_dic)
        #     Poly[i].setCoeff(sub)

        # piece=[]
        # piece.append( (Poly[0](x-h),x<=h) )

        # for i in range(1,n):
        #     piece.append( (Poly[i](x-(i+1)*h),x<=(i+1)*h) )

        # self.P=sp.Piecewise(*piece)

        # piece=[]
        # piece.append( (Poly[0].d(x-h),x<=h) )

        # for i in range(1,n):
        #     piece.append( (Poly[i].d(x-(i+1)*h),x<=(i+1)*h) )

        # self.Pd=sp.Piecewise(*piece)

        # piece=[]
        # piece.append( (Poly[0].dd(x-h),x<=h) )

        # for i in range(1,n):
        #     piece.append( (Poly[i].dd(x-(i+1)*h),x<=(i+1)*h) )

        # self.Pdd=sp.Piecewise(*piece)