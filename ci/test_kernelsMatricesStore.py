
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.kernelsMatricesStore

def test_eval_IndPointsLocsKMS():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtz = [torch.from_numpy(mat['Ktz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKttDiag = torch.from_numpy(mat['Ktt']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices_fromSpikes.mat")
    mat = loadmat(dataFilename)
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    leasKtz_spikes = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKttDiag_spikes = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}

    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsKMS.setEpsilon(epsilon=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    indPointsLocsKMS.buildKernelsMatrices()

    Kzz = indPointsLocsKMS.getKzz()
    for k in range(len(Kzz)):
        error = math.sqrt(((Kzz[k]-leasKzz[k])**2).flatten().mean())
        assert(error<tol)

    '''
    Kzzi = indPointsLocsKMS.getKzzi()
    for k in range(len(Kzzi)):
        error = math.sqrt(((Kzzi[k]-leasKzzi[k])**2).flatten().mean())
        assert(error<tolKzzi)
    '''

def test_eval_IndPointsLocsAndAllTimesKMS():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    t = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtz = [torch.from_numpy(mat['Ktz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKttDiag = torch.from_numpy(mat['Ktt']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices_fromSpikes.mat")
    mat = loadmat(dataFilename)
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    leasKtz_spikes = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKttDiag_spikes = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}

    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndAllTimesKMS.setTimes(times=t)
    indPointsLocsAndAllTimesKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsAndAllTimesKMS.buildKernelsMatrices()

    Ktz_allTimes = indPointsLocsAndAllTimesKMS.getKtz()
    for k in range(len(Ktz_allTimes)):
        error = math.sqrt(((Ktz_allTimes[k]-leasKtz[k])**2).flatten().mean())
        assert(error<tol)

    KttDiag_allTimes = indPointsLocsAndAllTimesKMS.getKttDiag()
    error = math.sqrt(((KttDiag_allTimes-leasKttDiag)**2).flatten().mean())
    assert(error<tol)

def test_eval_IndPointsLocsAndAssocTimesKMS():
    tol = 1e-5
    tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    t = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzz = [torch.from_numpy(mat['Kzz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKzzi = [torch.from_numpy(mat['Kzzi'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKtz = [torch.from_numpy(mat['Ktz'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    leasKttDiag = torch.from_numpy(mat['Ktt']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernelsParams0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
            kernels[k] = stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices_fromSpikes.mat")
    mat = loadmat(dataFilename)
    Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
    leasKtz_spikes = [[torch.from_numpy(mat['Ktz'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]
    leasKttDiag_spikes = [[torch.from_numpy(mat['Ktt'][i,j]).type(torch.DoubleTensor) for j in range(nTrials)] for i in range(nLatents)]

    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}

    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    indPointsLocsAndAssocTimesKMS.setKernels(kernels=kernels)
    indPointsLocsAndAssocTimesKMS.setTimes(times=Y)
    indPointsLocsAndAssocTimesKMS.setInitialParams(initialParams=kmsParams0)
    indPointsLocsAndAssocTimesKMS.buildKernelsMatrices()

    Ktz_associatedTimes = indPointsLocsAndAssocTimesKMS.getKtz()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((Ktz_associatedTimes[k][tr]-leasKtz_spikes[k][tr])**2).flatten().mean())
        assert(error<tol)

    KttDiag_associatedTimes = indPointsLocsAndAssocTimesKMS.getKttDiag()
    for k in range(nLatents):
        for tr in range(nTrials):
            error = math.sqrt(((KttDiag_associatedTimes[k][tr]-leasKttDiag_spikes[k][tr])**2).flatten().mean())
            assert(error<tol)

if __name__=='__main__':
    test_eval_IndPointsLocsKMS()
    test_eval_IndPointsLocsAndAllTimesKMS()
    test_eval_IndPointsLocsAndAssocTimesKMS()
