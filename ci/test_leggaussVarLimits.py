
import pdb
import sys
import os
from scipy.io import loadmat
import torch
sys.path.append("../src")
from myMath.utils import leggaussVarLimits

def test_leggaussVarLimits():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/legquad.mat")
    mat = loadmat(dataFilename)
    n = mat["n"][0][0]
    a = mat["a"][0][0]
    b = mat["b"][0][0]

    # Matlab's points are in decreasing order while Pythons ones are in increasing order
    # This is what I should be able to do
    # x = torch.from_numpy(mat["x"]).type(torch.DoubleTensor)
    # x = x[::-1]
    # But torch does not support negative steps in slices
    # begin workaround
    x_tmp = torch.from_numpy(mat["x"][0,:]).type(torch.DoubleTensor)
    x = torch.empty(x_tmp.shape)
    for i in range(len(x)):
        x[i] = x_tmp[len(x_tmp)-1-i]
    # end workaround

    # Matlab's points are in decreasing order while Pythons ones are in increasing order
    # This is what I should be able to do
    # w = torch.from_numpy(mat["w"]).type(torch.DoubleTensor)
    # w = w[::-1]
    # But torch does not support negative steps in slices
    # begin workaround
    w_tmp = torch.from_numpy(mat["w"][0,:]).type(torch.DoubleTensor)
    w = torch.empty(w_tmp.shape)
    for i in range(len(w)):
        w[i] = w_tmp[len(w_tmp)-1-i]
    # end workaround

    px, pw = leggaussVarLimits(n=n, a=a, b=b, dtype=torch.double)
    xDiff = torch.mean((x-px)**2)
    assert(xDiff<tol)
    wDiff = torch.mean((w-pw)**2)
    assert(wDiff<tol)

if __name__=="__main__":
    test_leggaussVarLimits()
