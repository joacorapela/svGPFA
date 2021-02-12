import pdb
import sys
import os
from scipy.io import loadmat
import torch
import numpy as np
import scipy.optimize
sys.path.append("../src")
import stats.kernels
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound
import utils.svGPFA.initUtils

def test_get_flattened_params():
    nTrials = 2
    nIndPoints = [2, 2, 2]

    nLatents = len(nIndPoints)
    qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
    srQSigma0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
    initialParams = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}

    true_flattened_params = []
    for k in range(nLatents):
        true_flattened_params.extend(qMu0[k].flatten().tolist())
    for k in range(nLatents):
        true_flattened_params.extend(srQSigma0Vecs[k].flatten().tolist())

    svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    svPosteriorOnIndPoints.setInitialParams(initialParams=initialParams)
    flattened_params = svPosteriorOnIndPoints.get_flattened_params()

    assert(flattened_params==true_flattened_params)

def test_set_flattened_params():
    nTrials = 2
    nIndPoints = [2, 2, 2]

    nLatents = len(nIndPoints)

    qMu0_1 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
    srQSigma0Vecs_1 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
    initialParams_1 = {"qMu0": qMu0_1, "srQSigma0Vecs": srQSigma0Vecs_1}
    svPosteriorOnIndPoints_1 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    svPosteriorOnIndPoints_1.setInitialParams(initialParams=initialParams_1)
    flattened_params_1 = svPosteriorOnIndPoints_1.get_flattened_params()

    qMu0_2 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
    srQSigma0Vecs_2 = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
    initialParams_2 = {"qMu0": qMu0_2, "srQSigma0Vecs": srQSigma0Vecs_2}
    svPosteriorOnIndPoints_2 = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    svPosteriorOnIndPoints_2.setInitialParams(initialParams=initialParams_2)
    svPosteriorOnIndPoints_2.set_params_from_flattened(flattened_params=flattened_params_1)
    flattened_params_2 = svPosteriorOnIndPoints_2.get_flattened_params()

    assert(flattened_params_1==flattened_params_2)

def test_set_params_requires_grad():
    nTrials = 2
    nIndPoints = [2, 2, 2]

    nLatents = len(nIndPoints)

    qMu0 = [torch.rand((nTrials, nIndPoints[k], 1), dtype=torch.double) for k in range(nLatents)]
    srQSigma0Vecs = [torch.rand((nTrials, int(((nIndPoints[k]+1)*nIndPoints[k])/2), 1), dtype=torch.double) for k in range(nLatents)]
    initialParams = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    svPosteriorOnIndPoints = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    svPosteriorOnIndPoints.setInitialParams(initialParams=initialParams)
    svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=True)
    params = svPosteriorOnIndPoints.getParams()
    for param in params:
        assert(param.requires_grad)

    svPosteriorOnIndPoints.set_params_requires_grad(requires_grad=False)
    params = svPosteriorOnIndPoints.getParams()
    for param in params:
        assert(not param.requires_grad)

def test_buildQSigma():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/get_full_from_lowplusdiag.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[0]
    nTrials = mat['q_sqrt'][(0,0)].shape[2]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    q_sigma = [torch.from_numpy(mat['q_sigma'][(0,k)]).permute(2,0,1) for k in range(nLatents)]
    qMu0 = [[] for i in range(nLatents)]

    params0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
    qU.setInitialParams(initialParams=params0)
    qSigma = qU.buildQSigma();

    error = torch.tensor([(qSigma[k]-q_sigma[k]).norm() for k in range(len(qSigma))]).sum()

    assert(error<tol)

def test_indPoints_grads():
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
        svlb.set_svPosteriorOnIndPoints_params_from_flattened(flattened_params=z.tolist())
        svlb.set_svPosteriorOnIndPoints_params_requires_grad(requires_grad=True)
        value = -svlb.eval()
        return value

    def value_func(z):
        # pdb.set_trace()
        value = eval_func(z=z)
        return value.item()

    def grad_func(z):
        # pdb.set_trace()
        value = eval_func(z=z)
        value.backward(retain_graph=True)
        grad_list = svlb.get_flattened_svPosteriorOnIndPoints_params_grad()
        grad = np.array(grad_list)
        return grad

    x0 = np.array(svlb.get_flattened_svPosteriorOnIndPoints_params())
    err = scipy.optimize.check_grad(func=value_func, grad=grad_func, x0=x0)
    assert(err<tol)
    pdb.set_trace()

if __name__=="__main__":
    # test_get_flattened_params()
    # test_set_flattened_params()
    test_set_params_requires_grad()
    # test_buildQSigma()
    test_indPoints_grads()
