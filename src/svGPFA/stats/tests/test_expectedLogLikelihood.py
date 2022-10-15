
import sys
import os
import math
from scipy.io import loadmat
import numpy as np
import torch
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.svPosteriorOnIndPoints
import svGPFA.stats.svPosteriorOnLatents
import svGPFA.stats.svEmbedding
import svGPFA.stats.expectedLogLikelihood

def test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink():
    tol = 3e-4
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Elik = torch.from_numpy(mat['Elik'])
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
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qKParams0 = {"posterior_on_ind_points": qUParams0,
                 "kernels_matrices_store": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initial_params = {"posterior_on_latents": qKParams0,
                     "embedding": qHParams0}
    eLLCalculationParams = {"leg_quad_points": legQuadPoints,
                            "leg_quad_weights": legQuadWeights}

    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(svEmbeddingAllTimes=qHAllTimes,
                                 svEmbeddingAssocTimes=qHAssocTimes)

    eLL.setKernels(kernels=kernels)
    eLL.setInitialParams(initial_params=initial_params)
    eLL.setMeasurements(measurements=YNonStacked)
    eLL.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    eLL.setPriorCovRegParam(priorCovRegParam=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    eLL.buildKernelsMatrices()
    sELL = eLL.evalSumAcrossTrialsAndNeurons()

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

def test_evalSumAcrossTrialsAndNeurons_pointProcessQuad():
    tol = 3e-4
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    t = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1) for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor)
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze()
    hermQuadPoints = torch.from_numpy(mat['xxHerm']).type(torch.DoubleTensor)
    hermQuadWeights = torch.from_numpy(mat['wwHerm']).type(torch.DoubleTensor)
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1)
    Elik = torch.from_numpy(mat['Elik'])
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
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
                                              float(hprs[k,0][1])],
                                             dtype=torch.double)
        elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
            kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
                                             dtype=torch.double)
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qKParams0 = {"posterior_on_ind_points": qUParams0,
                 "kernels_matrices_store": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initial_params = {"posterior_on_latents": qKParams0,
                     "embedding": qHParams0}
    eLLCalculationParams = {"hermQuadPoints": hermQuadPoints,
                            "hermQuadWeights": hermQuadWeights,
                            "leg_quad_points": legQuadPoints,
                            "leg_quad_weights": legQuadWeights}


    qU = svGPFA.stats.svPosteriorOnIndPoints.SVPosteriorOnIndPointsChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndAllTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
    indPointsLocsAndAssocTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
    qKAllTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
    qKAssocTimes = svGPFA.stats.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
    qHAllTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAllTimes(svPosteriorOnLatents=qKAllTimes)
    qHAssocTimes = svGPFA.stats.svEmbedding.LinearSVEmbeddingAssocTimes(svPosteriorOnLatents=
                                               qKAssocTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLQuad(svEmbeddingAllTimes=qHAllTimes,
                              svEmbeddingAssocTimes=qHAssocTimes,
                              linkFunction=torch.exp)

    eLL.setKernels(kernels=kernels)
    eLL.setInitialParams(initial_params=initial_params)
    eLL.setMeasurements(measurements=YNonStacked)
    eLL.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    eLL.setPriorCovRegParam(priorCovRegParam=1e-5) # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    eLL.buildKernelsMatrices()
    sELL = eLL.evalSumAcrossTrialsAndNeurons()

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

if __name__=="__main__":
    test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink()
    test_evalSumAcrossTrialsAndNeurons_pointProcessQuad()
