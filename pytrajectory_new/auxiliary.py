# IMPORTS
import log





class IntegChain(object):
    '''
    This class provides a representation of an integrator chain consisting of sympy symbols.
    
    For the elements :math:`(x_i)_{i=1,...,n}` the relation
    :math:`\dot{x}_i = x_{i+1}` applies.
    
    Parameters
    ----------
    
    lst : list
        Ordered list of elements for the integrator chain
    
    Attributes
    ----------
    
    elements : tuple
        Ordered list of all elements that are part of the integrator chain
    
    upper : sympy.Symbol
        Upper end of the integrator chain
    
    lower : sympy.Symbol
        Lower end of the integrator chain
    '''
    
    def __init__(self, lst):
        self.elements = tuple(lst)
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
        for elem in self.elements:#[::-1]:
            s += ' -> ' + elem.name
        return s[4:]


def findIntegratorChains(fi, x_sym):
    '''
    here comes the docstring...
    
    Returns
    -------
    
    list
        Found integrator chains.
    
    list
        Indices of the equations that have to be solved using collocation.
    '''
    
    n = len(x_sym)
    assert n == len(fi)
    
    chaindict = {}
    for i in xrange(len(fi)):
        # substitution because of sympy difference betw. 1.0 and 1
        if isinstance(fi[i], sp.Basic):
            fi[i] = fi[i].subs(1.0, 1)

        for xx in x_sym:
            if fi[i] == xx:
                chaindict[xx] = x_sym[i]

        for uu in u_sym:
            if fi[i] == uu:
                chaindict[uu] = x_sym[i]

    # chaindict looks like this:  {u_1 : x_2, x_4 : x_3, x_2 : x_1}
    # where x_4 = d/dt x_3 and so on

    # find upper ends of integrator chains
    uppers = []
    for vv in chaindict.values():
        if (not chaindict.has_key(vv)):
            uppers.append(vv)

    # create ordered lists that temporarily represent the integrator chains
    tmpchains = []

    # therefore we flip the dictionary to work our way through its keys
    # (former values)
    dictchain = {v:k for k,v in chaindict.items()}

    for var in uppers:
        tmpchain = []
        vv = var
        tmpchain.append(vv)

        while dictchain.has_key(vv):
            vv = dictchain[vv]
            tmpchain.append(vv)

        tmpchains.append(tmpchain)

    # create an integrator chain object for every temporary chain
    chains = []
    for lst in tmpchains:
        ic = IntegChain(lst)
        chains.append(ic)
        log.info("      --> found: " + str(ic), verb=3)
    
    # now we determine the equations that have to be solved by collocation
    # (--> lower ends of integrator chains)
    eqind = []

    if chains:
        # iterate over all integrator chains
        for ic in chains:
            # if lower end is a system variable
            # then its equation has to be solved
            if ic.lower.name.startswith('x'):
                idx = x_sym.index(ic.lower)
                eqind.append(idx)
        eqind.sort()
        
        # if every integrator chain ended with input variable
        if not eqind:
            eqind = range(n)
    else:
        # if integrator chains should not be used
        # then every equation has to be solved by collocation
        eqind = range(n)
    
    
    return chains, eqind
    
