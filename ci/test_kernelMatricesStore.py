
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
# from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import KernelMatricesStore
from kernels import PeriodicKernel, ExponentialQuadraticKernel

def test_eval():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.expanduser("data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    t = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtz = [torch.from_numpy(mat['Ktz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtt = torch.from_numpy(mat['Ktt']).type(torch.DoubleTensor).permute(2, 0, 1)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    kernels = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
            kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
        elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    dataFilename = os.path.expanduser("data/BuildKernelMatrices_fromSpikes.mat")
    mat = loadmat(dataFilename)
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    leasKtz_spikes = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKtt_spikes = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

    kernelsMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)

    Kzz = kernelsMatricesStore.getKzz()
    for k in range(len(Kzz)):
        error = math.sqrt(((Kzz[k]-leasKzz[k])**2).flatten().mean())
        assert(error<tol)

    Kzzi = kernelsMatricesStore.getKzzi()
    for k in range(len(Kzzi)):
        error = math.sqrt(((Kzzi[k]-leasKzzi[k])**2).flatten().mean())
        assert(error<tolKzzi)

    Ktz_allNeuronsAllTimes = kernelsMatricesStore.getKtz_allNeuronsAllTimes()
    for k in range(len(Ktz_allNeuronsAllTimes)):
        error = math.sqrt(((Ktz_allNeuronsAllTimes[k]-leasKtz[k])**2).flatten().mean())
        assert(error<tol)

    Ktt_allNeuronsAllTimes = kernelsMatricesStore.getKtt_allNeuronsAllTimes()
    error = math.sqrt(((Ktt_allNeuronsAllTimes-leasKtt)**2).flatten().mean())
    assert(error<tol)

    Ktz_allNeuronsAssociatedTimes = kernelsMatricesStore.getKtz_allNeuronsAssociatedTimes()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktz_allNeuronsAssociatedTimes[k][tr]-leasKtz_spikes[k][tr])**2).flatten().mean())
        assert(error<tol)

    Ktt_allNeuronsAssociatedTimes = kernelsMatricesStore.getKtt_allNeuronsAssociatedTimes()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktt_allNeuronsAssociatedTimes[k][tr]-leasKtt_spikes[k][tr])**2).flatten().mean())
            assert(error<tol)

if __name__=='__main__':
    test_eval()
