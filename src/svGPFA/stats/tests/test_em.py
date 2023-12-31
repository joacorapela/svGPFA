
import sys
import io
import os
import math
from scipy.io import loadmat
import numpy as np
import jax
import jax.numpy as jnp
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

jax.config.update("jax_enable_x64", True)

def test_emJAX__eval_func():
    tol = 3e-4
    reg_param = 1e-5
    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0 = [jax.device_put(mat['q_mu'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    C0 = jax.device_put(mat["C"].astype("float64"))
    b0 = jax.device_put(mat["b"].astype("float64").squeeze())
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2,0,1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2,0,1))
    obj = mat['obj'][0,0]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    qSigma0 = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
    qSigma0_chol_vecs = svGPFA.utils.miscUtils.getCholVecsFromCov(cov=qSigma0)

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = jax.device_put(YNonStacked_tmp[r,0][n,0][:,0]).astype("float64")

    YStacked, neuronForSpikeIndex = svGPFA.utils.miscUtils.stackSpikeTimes(spikeTimes=YNonStacked)

    linkFunction = jnp.exp

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=legQuadPoints)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=YStacked)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(
        posteriorOnLatents=qK)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(
        posteriorOnLatents=qK, neuronForSpikeIndex=neuronForSpikeIndex)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
        preIntensityQuadTimes=qHQuadTimes,
        preIntensitySpikesTimes=qHSpikesTimes,
        legQuadWeights=legQuadWeights,
    )
    klDiv = svGPFA.stats.klDivergence.KLDivergence(
        indPointsLocsKMS=indPointsLocsKMS)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)

    params0 = dict(
        variational_mean = qMu0,
        variational_chol_vecs = qSigma0_chol_vecs,
        C = C0,
        d = b0,
        kernels_params = kernels_params0,
        ind_points_locs = Z0,
    )
    em = svGPFA.stats.em.EM_JAX(model=svlb,
                                ind_points_locs_KMS=indPointsLocsKMS,
                                quad_times_KMS=quadTimesKMS,
                                spike_times_KMS=spikesTimesKMS,
                                reg_param=reg_param,
                               )

    lbEval = em._eval_func(params=params0)
    assert(abs(lbEval-obj)<tol)


# def test_eStep_pointProcess_PyTorch():
#     tol = 1e-5
#     yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Update_all_PointProcess_svGPFA.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['Z'])
#     nTrials = mat['Z'][0,0].shape[2]
#     qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
#     Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
#     b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
#     indPointsLocsKMSRegEpsilon = 1e-5
#     nLowerBound = mat['nLowerBound'][0,0]
#     legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
#     legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
#     kernelNames = mat["kernelNames"]
#     hprs = mat["hprs"]
# 
#     yMat = loadmat(yNonStackedFilename)
#     YNonStacked_tmp = yMat['YNonStacked']
#     nNeurons = YNonStacked_tmp[0,0].shape[0]
#     YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
#     for r in range(nTrials):
#         for n in range(nNeurons):
#             YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()
# 
#     linkFunction = torch.exp
# 
#     kernels = [[None] for k in range(nLatents)]
#     kernelsParams0 = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
#             kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
#                                               float(hprs[k,0][1])],
#                                              dtype=torch.double)
#         elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
#             kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
#                                              dtype=torch.double)
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
#     eLLCalculationParams = {"leg_quad_points": legQuadPoints,
#                   "leg_quad_weights": legQuadWeights}
# 
#     qU = svGPFA.stats.variationalDist.VariationalDistChol()
#     indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
#     indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
#     qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
#     qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
#     qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
#                                                qKSpikesTimes)
#     eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
#                                  preIntensitySpikesTimes=qHSpikesTimes)
#     klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
#                          variationalDist=qU)
#     svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
#     em = svGPFA.stats.em.EM_PyTorch()
# 
#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
# 
#     svlb.setKernels(kernels=kernels)
#     svlb.setInitialParams(initial_params=initial_params)
#     svlb.setMeasurements(measurements=YNonStacked)
#     svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
#     svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon)
#     svlb.buildKernelsMatrices()
#     logStream = io.StringIO()
# 
#     optim_params = {"max_iter": 100, "line_search_fn": "strong_wolfe"}
#     maxRes = em._eStep(model=svlb, optim_params=optim_params)
# 
#     assert(maxRes["lowerBound"]-(-nLowerBound)>0)

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

# def test_mStepModelParams_pointProcess_PyTorch():
#     yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/Mstep_Update_Iterative_PointProcess_svGPFA.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['Z'])
#     nTrials = mat['Z'][0,0].shape[2]
#     qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
#     Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     C0 = torch.from_numpy(mat["C0"]).type(torch.DoubleTensor).contiguous()
#     b0 = torch.from_numpy(mat["b0"]).type(torch.DoubleTensor).squeeze().contiguous()
#     indPointsLocsKMSRegEpsilon = 1e-5
#     nLowerBound = mat['nLowerBound'][0,0]
#     legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
#     legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
# 
#     yMat = loadmat(yNonStackedFilename)
#     YNonStacked_tmp = yMat['YNonStacked']
#     nNeurons = YNonStacked_tmp[0,0].shape[0]
#     YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
#     for r in range(nTrials):
#         for n in range(nNeurons):
#             YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()
# 
#     linkFunction = torch.exp
# 
#     kernelNames = mat["kernelNames"]
#     hprs = mat["hprs"]
#     kernels = [[None] for k in range(nLatents)]
#     kernelsParams0 = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
#             kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
#                                               float(hprs[k,0][1])],
#                                              dtype=torch.double)
#         elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
#             kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
#                                              dtype=torch.double)
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qU = svGPFA.stats.variationalDist.VariationalDistChol()
#     indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
#     indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
#     qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
#     qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
#     qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
#                                                qKSpikesTimes)
#     eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
#                                  preIntensitySpikesTimes=qHSpikesTimes)
#     klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
#                          variationalDist=qU)
#     svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
#     em = svGPFA.stats.em.EM_PyTorch()
# 
#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
#     eLLCalculationParams = {"leg_quad_points": legQuadPoints,
#                   "leg_quad_weights": legQuadWeights}
# 
#     svlb.setKernels(kernels=kernels)
#     svlb.setInitialParams(initial_params=initial_params)
#     svlb.setMeasurements(measurements=YNonStacked)
#     svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
#     svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
#     svlb.buildKernelsMatrices()
#     logStream = io.StringIO()
# 
#     optim_params = {"max_iter": 3000, "line_search_fn": "strong_wolfe"}
#     maxRes = em._mStepPreIntensity(model=svlb, optim_params=optim_params)
# 
#     assert(maxRes["lowerBound"]>-nLowerBound)

# def test_mStepKernelParams_pointProcess_PyTorch():
#     tol = 1e-5
#     yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/hyperMstep_Update.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['Z'])
#     nTrials = mat['Z'][0,0].shape[2]
#     qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
#     Z0 = [torch.from_numpy(mat['Z'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
#     b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
#     indPointsLocsKMSRegEpsilon = 1e-5
#     nLowerBound = mat['nLowerBound'][0,0]
#     legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
#     legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
# 
#     yMat = loadmat(yNonStackedFilename)
#     YNonStacked_tmp = yMat['YNonStacked']
#     nNeurons = YNonStacked_tmp[0,0].shape[0]
#     YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
#     for r in range(nTrials):
#         for n in range(nNeurons):
#             YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()
# 
#     linkFunction = torch.exp
# 
#     kernelNames = mat["kernelNames"]
#     hprs = mat["hprs0"]
#     kernels = [[None] for k in range(nLatents)]
#     kernelsParams0 = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
#             kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
#                                               float(hprs[k,0][1])],
#                                              dtype=torch.double)
#         elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
#             kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
#                                              dtype=torch.double)
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qU = svGPFA.stats.variationalDist.VariationalDistChol()
#     indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
#     indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
#     qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
#     qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
#     qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
#                                                qKSpikesTimes)
#     eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
#                                  preIntensitySpikesTimes=qHSpikesTimes)
#     klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
#                          variationalDist=qU)
#     svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
#     em = svGPFA.stats.em.EM_PyTorch()
# 
#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
#     eLLCalculationParams = {"leg_quad_points": legQuadPoints,
#                   "leg_quad_weights": legQuadWeights}
# 
#     svlb.setKernels(kernels=kernels)
#     svlb.setInitialParams(initial_params=initial_params)
#     svlb.setMeasurements(measurements=YNonStacked)
#     svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
#     svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
#     svlb.buildKernelsMatrices()
#     logStream = io.StringIO()
# 
#     optim_params = {"max_iter": 200, "line_search_fn": "strong_wolfe"}
#     maxRes = em._mStepKernels(model=svlb, optim_params=optim_params)
#     assert(maxRes["lowerBound"]>(-nLowerBound))

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

# def test_mStepIndPoints_pointProcess_PyTorch():
#     tol = 1e-5
#     yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/inducingPointsMstep_all.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['Z0'])
#     nTrials = mat['Z0'][0,0].shape[2]
#     qMu0 = [torch.from_numpy(mat['q_mu'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSVec0 = [torch.from_numpy(mat['q_sqrt'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     qSDiag0 = [torch.from_numpy(mat['q_diag'][(0,i)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
#     Z0 = [torch.from_numpy(mat['Z0'][(i,0)]).type(torch.DoubleTensor).permute(2,0,1).contiguous() for i in range(nLatents)]
#     C0 = torch.from_numpy(mat["C"]).type(torch.DoubleTensor).contiguous()
#     b0 = torch.from_numpy(mat["b"]).type(torch.DoubleTensor).squeeze().contiguous()
#     indPointsLocsKMSRegEpsilon = 1e-5
#     nLowerBound = mat['nLowerBound'][0,0]
#     legQuadPoints = torch.from_numpy(mat['ttQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
#     legQuadWeights = torch.from_numpy(mat['wwQuad']).type(torch.DoubleTensor).permute(2, 0, 1).contiguous()
# 
#     yMat = loadmat(yNonStackedFilename)
#     YNonStacked_tmp = yMat['YNonStacked']
#     nNeurons = YNonStacked_tmp[0,0].shape[0]
#     YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
#     for r in range(nTrials):
#         for n in range(nNeurons):
#             YNonStacked[r][n] = torch.from_numpy(YNonStacked_tmp[r,0][n,0][:,0]).type(torch.DoubleTensor).contiguous()
# 
#     linkFunction = torch.exp
# 
#     kernelNames = mat["kernelNames"]
#     hprs = mat["hprs"]
#     kernels = [[None] for k in range(nLatents)]
#     kernelsParams0 = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         if np.char.equal(kernelNames[0,k][0], "PeriodicKernel"):
#             kernels[k] = svGPFA.stats.kernels.PeriodicKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0]),
#                                               float(hprs[k,0][1])],
#                                              dtype=torch.double)
#         elif np.char.equal(kernelNames[0,k][0], "rbfKernel"):
#             kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=1.0)
#             kernelsParams0[k] = torch.tensor([float(hprs[k,0][0])],
#                                              dtype=torch.double)
#         else:
#             raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))
# 
#     qU = svGPFA.stats.variationalDist.VariationalDistChol()
#     indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol()
#     indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS()
#     qKQuadTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsQuadTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndQuadTimesKMS)
#     qKSpikesTimes = svGPFA.stats.posteriorOnLatents.PosteriorOnLatentsSpikesTimes(
#         variationalDist=qU,
#         indPointsLocsKMS=indPointsLocsKMS,
#         indPointsLocsAndTimesKMS=indPointsLocsAndSpikesTimesKMS)
#     qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(posteriorOnLatents=qKQuadTimes)
#     qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(posteriorOnLatents=
#                                                qKSpikesTimes)
#     eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(preIntensityQuadTimes=qHQuadTimes,
#                                  preIntensitySpikesTimes=qHSpikesTimes)
#     klDiv = svGPFA.stats.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
#                          variationalDist=qU)
#     svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)
#     em = svGPFA.stats.em.EM_PyTorch()
# 
#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
#     eLLCalculationParams = {"leg_quad_points": legQuadPoints,
#                   "leg_quad_weights": legQuadWeights}
# 
#     svlb.setKernels(kernels=kernels)
#     svlb.setInitialParams(initial_params=initial_params)
#     svlb.setMeasurements(measurements=YNonStacked)
#     svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
#     svlb.setPriorCovRegParam(priorCovRegParam=indPointsLocsKMSRegEpsilon) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
#     svlb.buildKernelsMatrices()
#     logStream = io.StringIO()
# 
#     optim_params = {"max_iter": 25, "line_search_fn": "strong_wolfe"}
#     maxRes = em._mStepIndPointsLocs(model=svlb, optim_params=optim_params)
#     assert(maxRes["lowerBound"]>(-nLowerBound))

def test_maximize_pointProcess_JAX():
    reg_param = 1e-5
    maxiter=3750
    max_stepsize = 200.0
    tol = 1e-5
    jit=False
    verbose=True

    yNonStackedFilename = os.path.join(os.path.dirname(__file__), "data/YNonStacked.mat")
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0 = [jax.device_put(mat['q_mu0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    # srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [jax.device_put(mat['Z0'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    C0 = jax.device_put(mat["C0"].astype("float64"))
    b0 = jax.device_put(mat["b0"].astype("float64"))
    indPointsLocsKMSRegEpsilon = 1e-2
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2, 0, 1))

    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = jax.device_put(YNonStacked_tmp[r,0][n,0][:,0].astype("float64"))

    YStacked, neuronForSpikeIndex = svGPFA.utils.miscUtils.stackSpikeTimes(spikeTimes=YNonStacked)

    linkFunction = jnp.exp

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
    leasLowerBound = mat['lowerBound'][0,0]

    qSigma0 = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
    qSigma0_chol_vecs = svGPFA.utils.miscUtils.getCholVecsFromCov(cov=qSigma0)

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=legQuadPoints)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=YStacked)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(
        posteriorOnLatents=qK)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(
        posteriorOnLatents=qK, neuronForSpikeIndex=neuronForSpikeIndex)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
        preIntensityQuadTimes=qHQuadTimes,
        preIntensitySpikesTimes=qHSpikesTimes,
        legQuadWeights=legQuadWeights,
    )
    klDiv = svGPFA.stats.klDivergence.KLDivergence(
        indPointsLocsKMS=indPointsLocsKMS)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)

    # svlb.setMeasurements(measurements=YNonStacked)
    # svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)

    params0 = dict(
        variational_mean = qMu0,
        variational_chol_vecs = qSigma0_chol_vecs,
        C = C0,
        d = b0,
        kernels_params = kernels_params0,
        ind_points_locs = Z0,
    )
    em = svGPFA.stats.em.EM_JAX(model=svlb,
                                ind_points_locs_KMS=indPointsLocsKMS,
                                quad_times_KMS=quadTimesKMS,
                                spike_times_KMS=spikesTimesKMS,
                                reg_param=reg_param,
                               )
    optim_params = dict(
        maxiter=maxiter,
        tol=tol,
        max_stepsize=max_stepsize,
        jit=jit,
        verbose=verbose,
    )

    start_time = time.time()
    res = em.maximize(params0=params0, optim_params=optim_params)
    elapsed_time = time.time() - start_time
    print(f"elapsed time={elapsed_time}")

    lower_bound = -res[1].value
    print(f"lower bound={lower_bound}")

    assert(lower_bound>leasLowerBound)

if __name__=='__main__':
    # test_emJAX__eval_func()
    # test_eStep_pointProcess_PyTorch() # passed
    # # test_eStep_poisson() # not tested
    # test_mStepModelParams_pointProcess_PyTorch()
    # test_mStepKernelParams_pointProcess_PyTorch() # passed
    # test_mStepIndPoints_pointProcess_PyTorch() # passed

    # t0 = time.perf_counter()
    test_maximize_pointProcess_JAX() # passed
    # elapsed = time.perf_counter()-t0
    # print(elapsed)

