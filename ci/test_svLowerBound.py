
import sys
import os
import pdb
import math
from scipy.io import loadmat
import numpy as np
import torch
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

def test_eval_pointProcess():
    tol = 3e-4
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

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
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    obj = mat['obj'][0,0]
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
    svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()

    lbEval = svlb.eval()

    assert(abs(lbEval+obj)<tol)

    # pdb.set_trace()

# def test_eval_poisson():
#     tol = 1e-5
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_svGPFA.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['q_mu'])
#     nTrials = mat['q_mu'][0,0].shape[2]
#     qMu = [torch.from_numpy(mat['q_mu'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSVec = [torch.from_numpy(mat['q_sqrt'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     qSDiag = [torch.from_numpy(mat['q_diag'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     t_tmp = torch.from_numpy(mat['tt']).type(torch.DoubleTensor).squeeze()
#     Z = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
#     Y = torch.from_numpy(mat['Y']).type(torch.DoubleTensor).permute(2,0,1)
#     C = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
#     b = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
#     hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
#     hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
#     binWidth = torch.from_numpy(mat['BinWidth'])
#     obj = mat['obj'][0,0]
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
#         if np.char.equal(kernelNames[0,k][0], 'PeriodicKernel'):
#             kernels[k] = PeriodicKernel(scale=1.0, lengthScale=float(hprs[k,0][0]), period=float(hprs[k,0][1]))
#         elif np.char.equal(kernelNames[0,k][0], 'rbfKernel'):
#             kernels[k] = ExponentialQuadraticKernel(scale=1.0, lengthScale=float(hprs[k,0][0]))
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,dtype=torch.uint8))
#     kernelsMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
#     qH = ApproxPosteriorForHForAllNeuronsAllTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
# 
#     eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsAllTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     lbEval = svlb.eval()
# 
#     assert(abs(lbEval+obj)<tol)
# 
if __name__=='__main__':
    test_eval_pointProcess()
    # test_eval_poisson()
