
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
sys.path.append("../src")
from stats.kernels import PeriodicKernel, ExponentialQuadraticKernel
from stats.svGPFA.kernelsMatricesStore import IndPointsLocsKMS, IndPointsLocsAndAllTimesKMS, IndPointsLocsAndAssocTimesKMS

def test_get_flattened_indPointsLocs():
    true_indPointsLocs = [torch.tensor([10.0, 20.0], dtype=torch.double),
                          torch.tensor([30.0, 40.0, 50.0], dtype=torch.double)]

    true_flattened_indPointsLocs = true_indPointsLocs[0].tolist() + true_indPointsLocs[1].tolist()

    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=true_indPointsLocs)
    flattened_indPointsLocs = indPointsLocsKMS.get_flattened_indPointsLocs()

    assert(true_flattened_indPointsLocs==flattened_indPointsLocs)

def test_set_indPointsLocs_from_flattened():
    initial_indPointsLocs = [torch.tensor([1.0, 2.0], dtype=torch.double),
                             torch.tensor([3.0, 4.0, 5.0], dtype=torch.double)]
    true_indPointsLocs = [torch.tensor([10.0, 20.0], dtype=torch.double),
                          torch.tensor([30.0, 40.0, 50.0], dtype=torch.double)]

    true_flattened_indPointsLocs = true_indPointsLocs[0].tolist() + true_indPointsLocs[1].tolist()

    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=initial_indPointsLocs)
    indPointsLocsKMS.set_indPointsLocs_from_flattened(flattened_params=true_flattened_indPointsLocs)
    indPointsLocs = indPointsLocsKMS.getIndPointsLocs()

    for k in range(len(indPointsLocs)):
        for i in range(len(indPointsLocs[k])):
            assert(indPointsLocs[k][i]==true_indPointsLocs[k][i])

def test_set_indPointsLocs_requires_grad():
    initial_indPointsLocs = [torch.tensor([1.0, 2.0], dtype=torch.double),
                             torch.tensor([3.0, 4.0, 5.0], dtype=torch.double)]

    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=initial_indPointsLocs)
    indPointsLocsKMS.set_indPointsLocs_requires_grad(requires_grad=True)
    for k in range(len(initial_indPointsLocs)):
        assert(initial_indPointsLocs[k].requires_grad)
    indPointsLocsKMS.set_indPointsLocs_requires_grad(requires_grad=False)
    for k in range(len(initial_indPointsLocs)):
        assert(not initial_indPointsLocs[k].requires_grad)

def test_get_flattened_kernels_params():
    initial_kernels_params = [torch.tensor([10.0, 20.0], dtype=torch.double), # priodic kernel lenghtscale and period
                              torch.tensor([30.0], dtype=torch.double) #exponential quadratic kernel lengthscale 
                             ]
    true_kernels_params = [torch.tensor([1.0, 2.0], dtype=torch.double), # priodic kernel lenghtscale and period
                           torch.tensor([3.0], dtype=torch.double) #exponential quadratic kernel lengthscale
                           ]

    true_flattened_kernels_params = true_kernels_params[0].tolist() + true_kernels_params[1].tolist()

    periodicKernel = PeriodicKernel(scale=1.0)
    periodicKernel.setParams(params=initial_kernels_params[0])

    exponentialQuadraticKernel = ExponentialQuadraticKernel(scale=1.0)
    exponentialQuadraticKernel.setParams(params=initial_kernels_params[1])

    kernels = [periodicKernel, exponentialQuadraticKernel]
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setKernelsParams(kernelsParams=true_kernels_params)
    flattened_kernels_params = indPointsLocsKMS.get_flattened_kernels_params()

    assert(true_flattened_kernels_params==flattened_kernels_params)

def test_set_kernels_params_from_flattened():
    initial_kernels_params = [torch.tensor([10.0, 20.0], dtype=torch.double), # priodic kernel lenghtscale and period
                              torch.tensor([30.0], dtype=torch.double) #exponential quadratic kernel lengthscale 
                             ]
    true_kernels_params = [torch.tensor([1.0, 2.0], dtype=torch.double), # priodic kernel lenghtscale and period
                           torch.tensor([3.0], dtype=torch.double) #exponential quadratic kernel lengthscale
                           ]

    true_flattened_kernels_params = true_kernels_params[0].tolist() + true_kernels_params[1].tolist()

    periodicKernel = PeriodicKernel(scale=1.0)
    periodicKernel.setParams(params=initial_kernels_params[0])

    exponentialQuadraticKernel = ExponentialQuadraticKernel(scale=1.0)
    exponentialQuadraticKernel.setParams(params=initial_kernels_params[1])

    kernels = [periodicKernel, exponentialQuadraticKernel]
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.set_kernels_params_from_flattened(flattened_params=true_flattened_kernels_params)
    flattened_kernels_params = indPointsLocsKMS.get_flattened_kernels_params()

    assert(true_flattened_kernels_params==flattened_kernels_params)

def test_set_kernels_params_requires_grad():
    initial_kernels_params = [torch.tensor([10.0, 20.0], dtype=torch.double), # priodic kernel lenghtscale and period
                              torch.tensor([30.0], dtype=torch.double) #exponential quadratic kernel lengthscale 
                             ]
    periodicKernel = PeriodicKernel(scale=1.0)
    periodicKernel.setParams(params=initial_kernels_params[0])

    exponentialQuadraticKernel = ExponentialQuadraticKernel(scale=1.0)
    exponentialQuadraticKernel.setParams(params=initial_kernels_params[1])

    kernels = [periodicKernel, exponentialQuadraticKernel]
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setKernelsParams(kernelsParams=initial_kernels_params)

    kernelsParams = indPointsLocsKMS.getKernelsParams()

    for i in range(len(kernelsParams)):
        assert(not kernelsParams[i].requires_grad)

    indPointsLocsKMS.set_kernels_params_requires_grad(requires_grad=True)
    for i in range(len(kernelsParams)):
        assert(kernelsParams[i].requires_grad)

    indPointsLocsKMS.set_kernels_params_requires_grad(requires_grad=False)
    for i in range(len(kernelsParams)):
        assert(not kernelsParams[i].requires_grad)

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
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
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

    indPointsLocsKMS = IndPointsLocsKMS()
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
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
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

    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
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
            kernels[k] = PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]), 
                                              float(hprs[k,0][1])], 
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = ExponentialQuadraticKernel(scale=1.0)
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

    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
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
    # test_eval_IndPointsLocsKMS()
    # test_eval_IndPointsLocsAndAllTimesKMS()
    # test_eval_IndPointsLocsAndAssocTimesKMS()
    # test_get_flattened_kernels_params()
    # test_set_kernels_params_from_flattened()
    # test_set_kernels_params_requires_grad()
    test_get_flattened_indPointsLocs()
    test_set_indPointsLocs_from_flattened()
    test_set_indPointsLocs_requires_grad()
