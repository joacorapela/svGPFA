
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import scipy.optimize
import torch
sys.path.append("../src")
from stats.kernels import PeriodicKernel, ExponentialQuadraticKernel
from stats.svGPFA.kernelsMatricesStore import IndPointsLocsKMS, IndPointsLocsAndAllTimesKMS, IndPointsLocsAndAssocTimesKMS
import utils.svGPFA.initUtils
import stats.kernels
import stats.svGPFA.kernelsMatricesStore
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound


def test_get_flattened_indPointsLocs():
    true_indPointsLocs = [torch.tensor([10.0, 20.0], dtype=torch.double),
                          torch.tensor([30.0, 40.0, 50.0], dtype=torch.double)]

    true_flattened_indPointsLocs = torch.cat((true_indPointsLocs[0], true_indPointsLocs[1]))

    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setIndPointsLocs(indPointsLocs=true_indPointsLocs)
    flattened_indPointsLocs = indPointsLocsKMS.get_flattened_indPointsLocs()

    assert(torch.all(torch.eq(true_flattened_indPointsLocs, flattened_indPointsLocs)))

def test_set_indPointsLocs_from_flattened():
    initial_indPointsLocs = [torch.tensor([1.0, 2.0], dtype=torch.double),
                             torch.tensor([3.0, 4.0, 5.0], dtype=torch.double)]
    true_indPointsLocs = [torch.tensor([10.0, 20.0], dtype=torch.double),
                          torch.tensor([30.0, 40.0, 50.0], dtype=torch.double)]

    true_flattened_indPointsLocs = torch.cat((true_indPointsLocs[0], true_indPointsLocs[1]))

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

    true_flattened_kernels_params = torch.cat((true_kernels_params[0], true_kernels_params[1]))
    periodicKernel = PeriodicKernel(scale=1.0)
    periodicKernel.setParams(params=initial_kernels_params[0])

    exponentialQuadraticKernel = ExponentialQuadraticKernel(scale=1.0)
    exponentialQuadraticKernel.setParams(params=initial_kernels_params[1])

    kernels = [periodicKernel, exponentialQuadraticKernel]
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.setKernelsParams(kernelsParams=true_kernels_params)
    flattened_kernels_params = indPointsLocsKMS.get_flattened_kernels_params()

    assert(torch.all(torch.eq(true_flattened_kernels_params, flattened_kernels_params)))

def test_set_kernels_params_from_flattened():
    initial_kernels_params = [torch.tensor([10.0, 20.0], dtype=torch.double), # priodic kernel lenghtscale and period
                              torch.tensor([30.0], dtype=torch.double) #exponential quadratic kernel lengthscale 
                             ]
    true_kernels_params = [torch.tensor([1.0, 2.0], dtype=torch.double), # priodic kernel lenghtscale and period
                           torch.tensor([3.0], dtype=torch.double) #exponential quadratic kernel lengthscale
                           ]

    true_flattened_kernels_params = torch.cat((true_kernels_params[0], true_kernels_params[1]))

    periodicKernel = PeriodicKernel(scale=1.0)
    periodicKernel.setParams(params=initial_kernels_params[0])

    exponentialQuadraticKernel = ExponentialQuadraticKernel(scale=1.0)
    exponentialQuadraticKernel.setParams(params=initial_kernels_params[1])

    kernels = [periodicKernel, exponentialQuadraticKernel]
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsKMS.setKernels(kernels=kernels)
    indPointsLocsKMS.set_kernels_params_from_flattened(flattened_params=true_flattened_kernels_params)
    flattened_kernels_params = indPointsLocsKMS.get_flattened_kernels_params()

    assert(torch.all(torch.eq(true_flattened_kernels_params, flattened_kernels_params)))

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

def test_kernels_grads():
    tol = 2e-3
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-2
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).contiguous().type(torch.DoubleTensor)

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
    leasLowerBound = mat['lowerBound'][0,0]
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

    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    svlb.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    optimParams = {"emMaxIter":3,
                   #
                   "eStepEstimate": True,
                   "eStepMaxIter":20,
                   "eStepTol":1e-2,
                   "eStepLR":1e-2,
                   "eStepLineSearchFn": "strong_wolfe",
                   "eStepNIterDisplay":1,
                   #
                   "mStepEmbeddingEstimate": True,
                   "mStepEmbeddingMaxIter":20,
                   "mStepEmbeddingTol":1e-2,
                   "mStepEmbeddingLR":1e-3,
                   "mStepEmbeddingLineSearchFn": "strong_wolfe",
                   "mStepEmbeddingNIterDisplay":1,
                   #
                   "mStepKernelsEstimate": True,
                   "mStepKernelsMaxIter":20,
                   "mStepKernelsTol":1e-2,
                   "mStepKernelsLR":1e-4,
                   "mStepKernelsLineSearchFn": "strong_wolfe",
                   "mStepKernelsNIterDisplay":1,
                   #
                   "mStepIndPointsEstimate": True,
                   "mStepIndPointsMaxIter":20,
                   "mStepIndPointsTol":1e-2,
                   "mStepIndPointsLR":1e-3,
                   "mStepIndPointsLineSearchFn": "strong_wolfe",
                   "mStepIndPointsNIterDisplay":1,
                   #
                   "verbose": True}
    svlb.setInitialParamsAndData( measurements=YNonStacked, initialParams=initialParams, quadParams=quadParams, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    def eval_func(z):
        # pdb.set_trace()
        svlb.set_kernels_params_from_flattened(flattened_params=z)
        svlb.set_kernels_params_requires_grad(requires_grad=True)
        svlb.buildKernelsMatrices()
        value = -svlb.eval()
        return value

    def value_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.tensor(z))
        return value.item()

    def grad_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.tensor(z))
        flattened_params = svlb.get_flattened_kernels_params()
        value.backward(retain_graph=True)
        grad = svlb.get_flattened_kernels_params_grad()
        grad_numpy = grad.numpy()
        return grad_numpy

    x0 = svlb.get_flattened_kernels_params().numpy()
    err = scipy.optimize.check_grad(func=value_func, grad=grad_func, x0=x0)
    assert(err<tol)

def test_indPointsLocs_grads():
    tol = 2e-3
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-2
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).contiguous().type(torch.DoubleTensor)

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
    leasLowerBound = mat['lowerBound'][0,0]
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

    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    svlb.setKernels(kernels=kernels)

    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    optimParams = {"emMaxIter":3,
                   #
                   "eStepEstimate": True,
                   "eStepMaxIter":20,
                   "eStepTol":1e-2,
                   "eStepLR":1e-2,
                   "eStepLineSearchFn": "strong_wolfe",
                   "eStepNIterDisplay":1,
                   #
                   "mStepEmbeddingEstimate": True,
                   "mStepEmbeddingMaxIter":20,
                   "mStepEmbeddingTol":1e-2,
                   "mStepEmbeddingLR":1e-3,
                   "mStepEmbeddingLineSearchFn": "strong_wolfe",
                   "mStepEmbeddingNIterDisplay":1,
                   #
                   "mStepKernelsEstimate": True,
                   "mStepKernelsMaxIter":20,
                   "mStepKernelsTol":1e-2,
                   "mStepKernelsLR":1e-4,
                   "mStepKernelsLineSearchFn": "strong_wolfe",
                   "mStepKernelsNIterDisplay":1,
                   #
                   "mStepIndPointsEstimate": True,
                   "mStepIndPointsMaxIter":20,
                   "mStepIndPointsTol":1e-2,
                   "mStepIndPointsLR":1e-3,
                   "mStepIndPointsLineSearchFn": "strong_wolfe",
                   "mStepIndPointsNIterDisplay":1,
                   #
                   "verbose": True}
    svlb.setInitialParamsAndData( measurements=YNonStacked, initialParams=initialParams, quadParams=quadParams, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)

    def eval_func(z):
        # pdb.set_trace()
        svlb.set_indPointsLocs_from_flattened(flattened_params=z)
        svlb.set_indPointsLocs_requires_grad(requires_grad=True)
        svlb.buildKernelsMatrices()
        value = -svlb.eval()
        return value

    def value_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.tensor(z))
        return value.item()

    def grad_func(z):
        # pdb.set_trace()
        value = eval_func(z=torch.tensor(z))
        value.backward(retain_graph=True)
        grad = svlb.get_flattened_indPointsLocs_grad()
        grad_numpy = grad.numpy()
        return grad_numpy

    x0 = svlb.get_flattened_indPointsLocs().numpy()
    err = scipy.optimize.check_grad(func=value_func, grad=grad_func, x0=x0)
    assert(err<tol)
    # pdb.set_trace()

if __name__=='__main__':
    # test_eval_IndPointsLocsKMS()
    # test_eval_IndPointsLocsAndAllTimesKMS()
    # test_eval_IndPointsLocsAndAssocTimesKMS()
    # test_get_flattened_kernels_params()
    # test_set_kernels_params_from_flattened()
    # test_set_kernels_params_requires_grad()
    # test_get_flattened_indPointsLocs()
    # test_set_indPointsLocs_from_flattened()
    # test_set_indPointsLocs_requires_grad()
    test_kernels_grads()
    # test_indPointsLocs_grads()
