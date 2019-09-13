
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
# from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import KernelMatricesStoreAllNeuronsAssociatedTimes
from kernels import PeriodicKernel, ExponentialQuadraticKernel

def test_eval():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/BuildKernelMatrices_fromSpikes.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtz = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKtt = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

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

    kernelsMatricesStore = KernelMatricesStoreAllNeuronsAssociatedTimes(kernels=kernels, Z=Z, Y=Y)

    Ktz = kernelsMatricesStore.getKtz()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktz[k][tr]-leasKtz[k][tr])**2).flatten().mean())
        assert(error<tol)

    Ktt = kernelsMatricesStore.getKtt()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktt[k][tr]-leasKtt[k][tr])**2).flatten().mean())
            assert(error<tol)

    pdb.set_trace()

if __name__=='__main__':
    test_eval()
