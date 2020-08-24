
import sys
import io
import os
import pdb
import math
from scipy.io import loadmat
import torch
import numpy as np
import time
sys.path.append("../src")
import utils.svGPFA.initUtils
import stats.kernels
import stats.svGPFA.kernelsMatricesStore
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound
import stats.svGPFA.svEM

def test_eStep_pointProcess():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Update_all_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSRSigma0Vec = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    linkFunction = torch.exp

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

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

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
    svEM = stats.svGPFA.svEM.SVEM()

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    res = svEM._eStep(model=svlb, maxIter=1500, tol=1e-3, lr=1e-3,
                      lineSearchFn="strong_wolfe", verbose=True,
                      out=sys.stdout, nIterDisplay=1, logLock=None,
                      logStream=logStream, logStreamFN=None)

    assert(res["lowerBound"]-(-nLowerBound)>0)

    # pdb.set_trace()

# def test_eStep_poisson():
#     tol = 1e-5
#     verbose = True
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Update_all_svGPFA.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = mat['q_mu'].shape[1]
#     nTrials = mat['q_mu'][0,0].shape[2]
#     qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     t_tmp = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).squeeze()
#     Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1)
#     C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
#     b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
#     nLowerBound = mat['nLowerBound'][0,0]
#     hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
#     hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
#     binWidth = mat['BinWidth'][0][0]
# 
#     # t_tmp \in nQuad and we want t \in nTrials x nQuad x 1
#     t = torch.ger(input=torch.ones(nTrials, dtype=torch.double), vec2=t_tmp).unsqueeze(dim=2)
# 
#     linkFunction = torch.exp
# 
#     kernelNames = mat["kernelNames"]
#     hprs = mat["hprs"]
#     kernels = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#                 if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
#                                 kernels[k] = PeriodicKernel(scale=1.0,
#                                         lengthScale=float(hprs[k,0][0]),
#                                         period=float(hprs[k,0][1]))
#                 elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
#                     kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
#                 else:
#                     raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
# 
#     qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
#     kernelsMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
#     qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
#     eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     svEM = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelsMatricesStore=kernelsMatricesStore)
#     res = svEM._SparseVariationalEM__eStep(maxIter=1000, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=100)
# 
#     assert(res["lowerBound"]-(-nLowerBound)>0)
# 
#     # pdb.set_trace()

def test_mStepModelParams_pointProcess():
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Mstep_Update_Iterative_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSRSigma0Vec = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    linkFunction = torch.exp

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
    svEM = stats.svGPFA.svEM.SVEM()

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    res = svEM._mStepEmbedding(model=svlb, maxIter=3000, tol=1e-6, lr=1e-1,
                               lineSearchFn="strong_wolfe", verbose=True,
                               out=sys.stdout, nIterDisplay=1, logLock=None,
                               logStream=logStream, logStreamFN=None)
    assert(res["lowerBound"]>-nLowerBound)

    # pdb.set_trace()

def test_mStepKernelParams_pointProcess():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/hyperMstep_Update.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSRSigma0Vec = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
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
    svEM = stats.svGPFA.svEM.SVEM()

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    res = svEM._mStepKernels(model=svlb, maxIter=50, tol=1e-3, lr=1e-3,
                             lineSearchFn="strong_wolfe", verbose=True,
                             out=sys.stdout, nIterDisplay=1, logLock=None,
                             logStream=logStream, logStreamFN=None)

    assert(res["lowerBound"]>(-nLowerBound))

    # pdb.set_trace()

# def test_mStepKernelParams_poisson():
#     tol = 1e-5
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/hyperMstep_Update.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['Z'])
#     nTrials = mat['Z'][0,0].shape[2]
#     qMu = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSVec = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSDiag = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
#     Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     Y = [torch.from_numpy(mat['Y'][tr,0]).type(torch.DoubleTensor) for tr in range(nTrials)]
#     C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
#     b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
#     index = [torch.from_numpy(mat['index'][i,0][:,0]).type(torch.ByteTensor) for i in range(nTrials)]
#     nLowerBound = mat['nLowerBound'][0,0]
#     hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
#     hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
#     legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
#     legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
# 
#     linkFunction = torch.exp
# 
#     kernelNames = mat["kernelNames"]
#     hprs0 = mat["hprs0"]
#     kernels = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
#             kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs0[k,0][0]), period=float(hprs0[k,0][1]))
#         elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
#             kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs0[k,0][0]))
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
#     kernelsMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
# 
#     qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
#     qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore, neuronForSpikeIndex=index)
# 
#     eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH_allNeuronsAllTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     svEM = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelsMatricesStore=kernelsMatricesStore)
#     res = svEM._SparseVariationalEM__mStepKernelParams(maxIter=50, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=10)
# 
#     assert(res["lowerBound"]>(-nLowerBound))
# 
#     # pdb.set_trace()

def test_mStepIndPoints_pointProcess():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/inducingPointsMstep_all.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSRSigma0Vec = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

    linkFunction = torch.exp

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
    svEM = stats.svGPFA.svEM.SVEM()

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    res = svEM._mStepIndPointsLocs(model=svlb, maxIter=10, tol=1e-3, lr=1e-3,
                               lineSearchFn="strong_wolfe", verbose=True,
                               out=sys.stdout, nIterDisplay=1, logLock=None,
                               logStream=logStream, logStreamFN=None)

    assert(res["lowerBound"]>(-nLowerBound))

    # pdb.set_trace()

def test_maximize_pointProcess():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")
    # yNonStackedFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/YNonStacked.mat")
    # dataFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/variationalEM.mat")
    # yNonStackedFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/YNonStacked.mat")
    # dataFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSRSigma0Vec = utils.svGPFA.initUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
    indPointsLocsKMSRegEpsilon = 1e-2
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor)

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
    svEM = stats.svGPFA.svEM.SVEM()

    qUParams0 = {"qMu0": qMu0, "qSRSigma0Vec": qSRSigma0Vec}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}
    '''
    optimParams = {"emMaxNIter":20, 
                   #
                   "eStepMaxNIter":100,
                   "eStepTol":1e-3,
                   "eStepLR":1e-3,
                   "eStepNIterDisplay":10,
                   #
                   "mStepModelParamsMaxNIter":100,
                   "mStepModelParamsTol":1e-3,
                   "mStepModelParamsLR":1e-3,
                   "mStepModelParamsNIterDisplay":10,
                   #
                   "mStepKernelParamsMaxNIter":100, 
                   "mStepKernelParamsTol":1e-3,
                   "mStepKernelParamsLR":1e-5,
                   "mStepKernelParamsNIterDisplay":10,
                   #
                   "mStepIndPointsMaxNIter":100,
                   "mStepIndPointsParamsTol":1e-3,
                   "mStepIndPointsLR":1e-3, 
                   "mStepIndPointsNIterDisplay":10}
    '''
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
    lowerBoundHist, elapsedTimeHist = svEM.maximize(
        model=svlb, measurements=YNonStacked,
        initialParams=initialParams, quadParams=quadParams,
        optimParams=optimParams,
        indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon, out=sys.stdout)
    assert(lowerBoundHist[-1]>leasLowerBound)

if __name__=='__main__':
    # test_eStep_pointProcess() # passed
    # # test_eStep_poisson() # not tested
    test_mStepModelParams_pointProcess() # passed
    test_mStepKernelParams_pointProcess() # passed
    test_mStepIndPoints_pointProcess() # passed

    t0 = time.perf_counter()
    test_maximize_pointProcess() # passed
    elapsed = time.perf_counter()-t0
    print(elapsed)

    pdb.set_trace()
