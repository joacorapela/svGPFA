import pdb
import torch
from numpy.polynomial.legendre import leggauss

def leggaussVarLimits(n, a, b, dtype=torch.double):
    """
    Computers n weights and points for Gauss-Legendre numerical integration of a
    function in the interval (a,b).
    """
    x, w = leggauss(deg=n)
    # reversing x and w to make them compatible with Lea's Matlab legquad function
    # x = x[::-1].copy()
    # w = w[::-1].copy()
    x = torch.from_numpy(x)
    w = torch.from_numpy(w)
    xVarLimits = (x*(b-a)+(b+a))/2
    wVarLimits = (b-a)/2*w
    return xVarLimits, wVarLimits
