
import sys
import io
import os
import math
from scipy.io import loadmat
import torch
import numpy as np
import time
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.variationalDist
import svGPFA.stats.posteriorOnLatents
import svGPFA.stats.preIntensity
import svGPFA.stats.expectedLogLikelihood
import svGPFA.stats.klDivergence
import svGPFA.stats.svLowerBound
import svGPFA.stats.em

def test_eStep_pointProcess_PyTorch():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Update_all_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()

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

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
                                               qKSpikesTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
                                 preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         variationalDist=qU)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    em = svGPFA.stats.em.EM_PyTorch()

    qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
    kmsParams0 = {"kernels_params0": kernelsParams0,
                  "inducing_points_locs0": Z0}
    qKParams0 = {"posterior_on_ind_points": qUParams0,
                 "kernels_matrices_store": kmsParams0}
    qHParams0 = {"C0": C0, "d0": b0}
    initial_params = {"posterior_on_latents": qKParams0,
                     "embedding": qHParams0}

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initial_params=initial_params)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon)
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    optim_params = {"max_iter": 100, "line_search_fn": "strong_wolfe"}
    maxRes = em._eStep(model=svlb, optim_params=optim_params)

    assert(maxRes["lowerBound"]-(-nLowerBound)>0)

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
#     qH = ApproxPosteriorForHForAllNeuronsQuadTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
#     eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsQuadTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     em = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelsMatricesStore=kernelsMatricesStore)
#     maxRes = em._SparseVariationalEM__eStep(maxIter=1000, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=100)
# 
#     assert(maxRes["lowerBound"]-(-nLowerBound)>0)

def test_mStepModelParams_pointProcess_PyTorch():
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Mstep_Update_Iterative_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
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

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
                                               qKSpikesTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
                                 preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         variationalDist=qU)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    em = svGPFA.stats.em.EM_PyTorch()

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

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initial_params=initial_params)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    optim_params = {"max_iter": 3000, "line_search_fn": "strong_wolfe"}
    maxRes = em._mStepPreIntensity(model=svlb, optim_params=optim_params)

    assert(maxRes["lowerBound"]>-nLowerBound)

def test_mStepKernelParams_pointProcess_PyTorch():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/hyperMstep_Update.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
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

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
                                               qKSpikesTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
                                 preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         variationalDist=qU)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    em = svGPFA.stats.em.EM_PyTorch()

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

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initial_params=initial_params)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    optim_params = {"max_iter": 200, "line_search_fn": "strong_wolfe"}
    maxRes = em._mStepKernels(model=svlb, optim_params=optim_params)
    assert(maxRes["lowerBound"]>(-nLowerBound))

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
#     qH_allNeuronsQuadTimes = ApproxPosteriorForHForAllNeuronsQuadTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
#     qH_allNeuronsAssociatedTimes = ApproxPosteriorForHForAllNeuronsAssociatedTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore, neuronForSpikeIndex=index)
# 
#     eLL = PointProcessExpectedLogLikelihood(approxPosteriorForHForAllNeuronsQuadTimes=qH_allNeuronsQuadTimes, approxPosteriorForHForAllNeuronsAssociatedTimes=qH_allNeuronsAssociatedTimes, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights, linkFunction=linkFunction)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     em = SparseVariationalEM(lowerBound=svlb, eLL=eLL, kernelsMatricesStore=kernelsMatricesStore)
#     maxRes = em._SparseVariationalEM__mStepKernelParams(maxIter=50, tol=1e-3, lr=1e-3, verbose=True, nIterDisplay=10)
# 
#     assert(maxRes["lowerBound"]>(-nLowerBound))

def test_mStepIndPoints_pointProcess_PyTorch():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/inducingPointsMstep_all.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
    b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
    indPointsLocsKMSRegEpsilon = 1e-5
    nLowerBound = mat['nLowerBound'][0,0]
    legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
    legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()

    linkFunction = torch.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
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

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
                                               qKSpikesTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
                                 preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         variationalDist=qU)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    em = svGPFA.stats.em.EM_PyTorch()

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

    svlb.setKernels(kernels=kernels)
    svlb.setInitialParams(initial_params=initial_params)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
    svlb.buildKernelsMatrices()
    logStream = io.StringIO()

    optim_params = {"max_iter": 25, "line_search_fn": "strong_wolfe"}
    maxRes = em._mStepIndPointsLocs(model=svlb, optim_params=optim_params)
    assert(maxRes["lowerBound"]>(-nLowerBound))

def test_maximize_pointProcess_PyTorch():
    tol = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [torch.from_numpy(mat['q_mu0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSVec0 = [torch.from_numpy(mat['q_sqrt0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    qSDiag0 = [torch.from_numpy(mat['q_diag0'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
    srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
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

    qU = svGPFA.stats.variationalDist.VariationalDistChol()
    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
    qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
    qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
        variationalDist=qU,
        indPointsLocsKMS=indPointsLocsKMS,
        indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
                                               qKSpikesTimes)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
                                 preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                         variationalDist=qU)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
    svlb.setKernels(kernels=kernels)
    em = svGPFA.stats.em.EM_PyTorch()

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

    optim_params = {"em_max_iter":4,
                   #
                   "estep_estimate": True,
                   "estep_optim_params": {
                       "max_iter": 20,
                       "line_search_fn": "strong_wolfe"
                   },
                   #
                   "mstep_embedding_estimate": True,
                   "mstep_embedding_optim_params": {
                       "max_iter": 20,
                       "line_search_fn": "strong_wolfe"
                   },
                   #
                   "mstep_kernels_estimate": True,
                   "mstep_kernels_optim_params": {
                       "max_iter": 20,
                       "line_search_fn": "strong_wolfe"
                   },
                   #
                   "mstep_indpointslocs_estimate": True,
                   "mstep_indpointslocs_optim_params": {
                       "max_iter": 20,
                       "line_search_fn": "strong_wolfe"
                   },
                   #
                   "verbose": True}
    svlb.setParamsAndData(
        measurements=YNonStacked,
        initial_params=initial_params,
        eLLCalculationParams=eLLCalculationParams,
        priorCovRegParam=indPointsLocsKMSRegEpsilon)
    lowerBoundHist, _, _, _ = em.maximizeInSteps(model=svlb, optim_params=optim_params)
    assert(lowerBoundHist[-1]>leasLowerBound)

if __name__=='__main__':
    # test_eStep_pointProcess_PyTorch() # passed
    # # test_eStep_poisson() # not tested
    # test_mStepModelParams_pointProcess_PyTorch()
    # test_mStepKernelParams_pointProcess_PyTorch() # passed
    # test_mStepIndPoints_pointProcess_PyTorch() # passed

    t0 = time.perf_counter()
    test_maximize_pointProcess_PyTorch() # passed
    elapsed = time.perf_counter()-t0
    print(elapsed)

