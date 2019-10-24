
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
# from inducingPointsPrior import InducingPointsPrior
from kernelMatricesStore import IndPointsLocsKMS, IndPointsLocsAndAllTimesKMS, IndPointsLocsAndAssocTimesKMS
from kernels import PeriodicKernel, ExponentialQuadraticKernel

def test_eval():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

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

    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices_fromSpikes.mat")
    mat = loadmat(dataFilename)
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    leasKtz_spikes = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKtt_spikes = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setIndPointsLocs(locs=Z)
    indPointsLocsKMS.buildKernelsMatrices()

    Kzz = indPointsLocsKMS.getKzz()
    for k in range(len(Kzz)):
        error = math.sqrt(((Kzz[k]-leasKzz[k])**2).flatten().mean())
        assert(error<tol)

    Kzzi = indPointsLocsKMS.getKzzi()
    for k in range(len(Kzzi)):
        error = math.sqrt(((Kzzi[k]-leasKzzi[k])**2).flatten().mean())
        assert(error<tolKzzi)

    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndAllTimesKMS.setIndPointsLocs(locs=Z)
    indPointsLocsAndAllTimesKMS.setTimes(times=t)
    indPointsLocsAndAllTimesKMS.buildKernelsMatrices()

    Ktz_allTimes = indPointsLocsAndAllTimesKMS.getKtz()
    for k in range(len(Ktz_allTimes)):
        error = math.sqrt(((Ktz_allTimes[k]-leasKtz[k])**2).flatten().mean())
        assert(error<tol)

    Ktt_allTimes = indPointsLocsAndAllTimesKMS.getKtt()
    error = math.sqrt(((Ktt_allTimes-leasKtt)**2).flatten().mean())
    assert(error<tol)

    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    indPointsLocsAndAssocTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndAssocTimesKMS.setIndPointsLocs(locs=Z)
    indPointsLocsAndAssocTimesKMS.setTimes(times=Y)
    indPointsLocsAndAssocTimesKMS.buildKernelsMatrices()

    Ktz_associatedTimes = indPointsLocsAndAssocTimesKMS.getKtz()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktz_associatedTimes[k][tr]-leasKtz_spikes[k][tr])**2).flatten().mean())
        assert(error<tol)

    Ktt_associatedTimes = indPointsLocsAndAssocTimesKMS.getKtt()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktt_associatedTimes[k][tr]-leasKtt_spikes[k][tr])**2).flatten().mean())
            assert(error<tol)

if __name__=='__main__':
    test_eval()
