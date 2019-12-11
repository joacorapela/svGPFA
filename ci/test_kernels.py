
import pdb
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
sys.path.append("../src")
from stats.kernels import ExponentialQuadraticKernel, PeriodicKernel

def test_exponentialQuadraticKernel():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/rbfKernel.mat")

    mat = loadmat(dataFilename)
    Z = torch.from_numpy(mat['X1']).type(torch.DoubleTensor).permute(2,0,1)
    leasK = torch.from_numpy(mat['G']).type(torch.DoubleTensor).permute(2,0,1)
    lengthScale = float(mat['lengthscale'][0,0])
    scale = 1.0
    params = [lengthScale]

    kernel = ExponentialQuadraticKernel(scale=scale) 
    kernel.setParams(params=params)

    K = kernel.buildKernelMatrix(X1=Z)

    error = math.sqrt(((K-leasK)**2).flatten().mean())

    assert(error<tol)

def test_exponentialQuadraticKernelDiag():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Kdiag_rbfKernel.mat")

    mat = loadmat(dataFilename)
    t = torch.from_numpy(mat['X1']).type(torch.DoubleTensor).permute(2,0,1)
    leasKDiag = torch.from_numpy(mat['Gdiag']).type(torch.DoubleTensor).permute(2,0,1)
    lengthScale = float(mat['lengthscale'][0,0])
    scale = float(mat['variance'][0,0])
    params = [lengthScale]

    kernel = ExponentialQuadraticKernel(scale=scale)
    kernel.setParams(params=params)

    KDiag = kernel.buildKernelMatrixDiag(X=t)

    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())

    assert(error<tol)

def test_periodicKernel():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/PeriodicKernel.mat")

    mat = loadmat(dataFilename)
    Z = torch.from_numpy(mat['X1']).type(torch.DoubleTensor).permute(2,0,1)
    leasK = torch.from_numpy(mat['G']).type(torch.DoubleTensor).permute(2,0,1)
    lengthScale = float(mat['lengthscale'][0,0])
    period = float(mat['period'][0,0])
    scale = 1.0
    params = [lengthScale, period]

    kernel = PeriodicKernel(scale=scale)
    kernel.setParams(params=params)

    K = kernel.buildKernelMatrix(X1=Z)

    error = math.sqrt(((K-leasK)**2).flatten().mean())

    assert(error<tol)

def test_periodicKernelDiag():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Kdiag_PeriodicKernel.mat")

    mat = loadmat(dataFilename)
    t = torch.from_numpy(mat['X1']).type(torch.DoubleTensor).permute(2,0,1)
    leasKDiag = torch.from_numpy(mat['Gdiag']).type(torch.DoubleTensor).permute(2,0,1)
    lengthScale = float(mat['lengthscale'][0,0])
    period = float(mat['period'][0,0])
    scale = float(mat['variance'][0,0])
    params = [lengthScale, period]

    kernel = PeriodicKernel(scale=scale)
    kernel.setParams(params=params)

    KDiag = kernel.buildKernelMatrixDiag(X=t)

    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())

    assert(error<tol)

if __name__=="__main__":
    test_exponentialQuadraticKernel()
    test_exponentialQuadraticKernelDiag()
    test_periodicKernel()
    test_periodicKernelDiag()

