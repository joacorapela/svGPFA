
import sys
import os
from scipy.io import loadmat
import torch
sys.path.append("../src")
import numericalMethods.utils

def test_leggaussVarLimits():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/legquad.mat")
    mat = loadmat(dataFilename)
    n = mat["n"][0][0]
    a = mat["a"][0][0]
    b = mat["b"][0][0]

    x = torch.flip(input=torch.from_numpy(mat["x"]).type(torch.DoubleTensor), dims=[1])
    w = torch.flip(input=torch.from_numpy(mat["w"]).type(torch.DoubleTensor), dims=[1])

    px, pw = numericalMethods.utils.leggaussVarLimits(n=n, a=a, b=b, dtype=torch.double)
    xDiff = torch.mean((x-px)**2)
    assert(xDiff<tol)
    wDiff = torch.mean((w-pw)**2)
    assert(wDiff<tol)

if __name__=="__main__":
    test_leggaussVarLimits()
