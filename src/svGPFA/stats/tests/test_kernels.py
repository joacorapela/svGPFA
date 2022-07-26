import sys
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
import svGPFA.stats.kernels

def test_exponentialQuadraticKernel():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/rbfKernel.mat")

    mat = loadmat(dataFilename)
    Z = torch.from_numpy(mat['X1']).type(torch.DoubleTensor).permute(2,0,1)
    leasK = torch.from_numpy(mat['G']).type(torch.DoubleTensor).permute(2,0,1)
    lengthScale = float(mat['lengthscale'][0,0])
    scale = 1.0
    params = torch.tensor([lengthScale])

    kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=scale) 
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
    params = torch.tensor([lengthScale])

    kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=scale)
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
    params = torch.tensor([lengthScale, period])

    kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale)
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
    params = torch.tensor([lengthScale, period])

    kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale)
    kernel.setParams(params=params)

    KDiag = kernel.buildKernelMatrixDiag(X=t)

    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())

    assert(error<tol)

if __name__=="__main__":
    test_exponentialQuadraticKernel()
    test_exponentialQuadraticKernelDiag()
    test_periodicKernel()
    test_periodicKernelDiag()

