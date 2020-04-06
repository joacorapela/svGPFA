
import sys
import os
import pdb
import math
from scipy.io import loadmat
import torch
import numpy as np
import time
sys.path.append("../src")
from stats.kernels import PeriodicKernel, ExponentialQuadraticKernel
from stats.svGPFA.kernelMatricesStore import IndPointsLocsKMS, \
        IndPointsLocsAndAllTimesKMS, IndPointsLocsAndAssocTimesKMS
from stats.svGPFA.svPosteriorOnIndPoints import SVPosteriorOnIndPoints
from stats.svGPFA.svPosteriorOnLatents import SVPosteriorOnLatentsAllTimes,\
        SVPosteriorOnLatentsAssocTimes
from stats.svGPFA.svEmbedding import LinearSVEmbeddingAllTimes, \
        LinearSVEmbeddingAssocTimes
from stats.svGPFA.expectedLogLikelihood import PointProcessELLExpLink, \
        PointProcessELLQuad
from stats.svGPFA.klDivergence import KLDivergence
from stats.svGPFA.svLowerBound import SVLowerBound
from stats.svGPFA.svEM import SVEM

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
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
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

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SVEM()

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.buildKernelsMatrices()

    res = svEM._eStep(model=svlb, maxNIter=1500, tol=1e-3, lr=1e-3,
                      verbose=True, nIterDisplay=100)

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
#     kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
#     qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
#     eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
#     klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     svEM = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelMatricesStore=kernelMatricesStore)
#     res = svEM._SparseVariationalEM__eStep(maxNIter=1000, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=100)
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
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SVEM()

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.buildKernelsMatrices()

    res = svEM._mStepModelParams(model=svlb, maxNIter=3000, tol=1e-6, lr=1e-1, verbose=True, nIterDisplay=1)

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
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SVEM()

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.buildKernelsMatrices()

    res = svEM._mStepKernelParams(model=svlb, maxNIter=50, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=10)

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
#     kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
# 
#     qH_allNeuronsAllTimes = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore)
#     qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelMatricesStore=kernelMatricesStore, neuronForSpikeIndex=index)
# 
#     eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH_allNeuronsAllTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
#     klDiv = KLDivergence(kernelMatricesStore=kernelMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     svEM = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelMatricesStore=kernelMatricesStore)
#     res = svEM._SparseVariationalEM__mStepKernelParams(maxNIter=50, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=10)
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
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
    svEM = SVEM()

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initialParams=initialParams)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setQuadParams(quadParams=quadParams)
    svlb.buildKernelsMatrices()

    res = svEM._mStepIndPoints(model=svlb, maxNIter=100, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=10)

    assert(res["lowerBound"]>(-nLowerBound))

    # pdb.set_trace()

def test_maximize_pointProcess():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")
    # yNonStackedFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/YNonStacked.mat")
    # dataFilename = os.path.expanduser("~/tmp/svGPFA/ci/data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze()
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

    qU = SVPosteriorOnIndPoints()
    indPointsLocsKMS = IndPointsLocsKMS()
    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)
    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         svPosteriorOnIndPoints=qU)
    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
    svlb.setKernels(kernels=kernels)
    svEM = SVEM()

    qUParams0 = {"qMu0": qMu0, "qSVec0": qSVec0, "qSDiag0": qSDiag0}
    qHParams0 = {"C0": C0, "d0": b0}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    initialParams = {"svPosteriorOnIndPoints": qUParams0,
                     "kernelsMatricesStore": kmsParams0,
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
    optimParams = {"emMaxNIter":30, 
                   #
                   "eStepMaxNIter":20,
                   "eStepTol":1e-2,
                   "eStepLR":1e-2,
                   "eStepNIterDisplay":1,
                   #
                   "mStepModelParamsMaxNIter":20,
                   "mStepModelParamsTol":1e-2,
                   "mStepModelParamsLR":1e-3,
                   "mStepModelParamsNIterDisplay":1,
                   #
                   "mStepKernelParamsMaxNIter":50, 
                   "mStepKernelParamsTol":1e-2,
                   "mStepKernelParamsLR":1e-4,
                   "mStepKernelParamsNIterDisplay":1,
                   #
                   "mStepIndPointsMaxNIter":80,
                   "mStepIndPointsParamsTol":1e-2,
                   "mStepIndPointsLR":1e-3, 
                   "mStepIndPointsNIterDisplay":1}
    lowerBoundHist, elapsedTimeHist = svEM.maximize(
        model=svlb, measurements=YNonStacked,
        initialParams=initialParams, quadParams=quadParams,
        optimParams=optimParams)
    assert(lowerBoundHist[-1]>leasLowerBound)

    # pdb.set_trace()

if __name__=='__main__':
    # test_eStep_pointProcess() # passed
    # # test_eStep_poisson() # not tested
    # test_mStepModelParams_pointProcess() # passed
    # test_mStepKernelParams_pointProcess() # passed
    # test_mStepIndPoints_pointProcess() # passed

    t0 = time.perf_counter()
    test_maximize_pointProcess() # passed
    elapsed = time.perf_counter()-t0
    print(elapsed)

    # pdb.set_trace()
