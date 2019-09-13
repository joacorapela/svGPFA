
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
# from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import KernelMatricesStoreAllNeuronsAllTimes
from kernels import PeriodicKernel, ExponentialQuadraticKernel

def test_eval():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatent = mat['Z'].shape[0]
    t = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    leasKzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    leasKzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    leasKtz = [torch.from_numpy(mat['Ktz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatent)]
    leasKtt = torch.from_numpy(mat['Ktt']).type(torch.DoubleTensor).permute(2, 0, 1)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatent)]
    for k in range(nLatent):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    kernelsMatricesStore = KernelMatricesStoreAllNeuronsAllTimes(kernels=kernels, Z=Z, t=t)

    Kzz = kernelsMatricesStore.getKzz()
    for k in range(len(Kzz)):
        error = math.sqrt(((Kzz[k]-leasKzz[k])**2).flatten().mean())
        assert(error<tol)

    Kzzi = kernelsMatricesStore.getKzzi()
    for k in range(len(Kzzi)):
        error = math.sqrt(((Kzzi[k]-leasKzzi[k])**2).flatten().mean())
        assert(error<tolKzzi)

    Ktz = kernelsMatricesStore.getKtz()
    for k in range(len(Ktz)):
        error = math.sqrt(((Ktz[k]-leasKtz[k])**2).flatten().mean())
        assert(error<tol)

    Ktt = kernelsMatricesStore.getKtt()
    error = math.sqrt(((Ktt-leasKtt)**2).flatten().mean())
    assert(error<tol)

    pdb.set_trace()

if __name__=='__main__':
    test_eval()
