
import sys
import os
import math
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.variationalDist
import svGPFA.stats.posteriorOnLatents
import svGPFA.stats.preIntensity
import svGPFA.stats.expectedLogLikelihood
import svGPFA.stats.klDivergence
import svGPFA.stats.svLowerBound

jax.config.update("jax_enable_x64", True)

def test_eval_pointProcess():
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
    t = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
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
    yMat = loadmat(yNonStackedFilename)
    YNonStacked_tmp = yMat['YNonStacked']
    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = jax.device_put(YNonStacked_tmp[r,0][n,0][:,0]).astype("float64")

    linkFunction = jnp.exp

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            scale = 1.0
            lengthscale = hprs[k,0][0].item() 
            period = hprs[k,0][1].item()
            lengthscaleScale = 1.0
            periodScale = 1.0
            kernels_params0[k] = {"scale": scale, "lengthscale": lengthscale,
                                  "lengthscaleScale": lengthscaleScale,
                                  "period": period, "periodScale": periodScale}
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            scale = 1.0
            lengthscale = hprs[k,0][0].item()
            lengthscaleScale = 1.0
            kernels_params0[k] = {"scale": scale, "lengthscale": lengthscale,
                                  "lengthscaleScale": lengthscaleScale}
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels)
    quadTimesKMS.setTimes(times=t)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qHQuadTimes = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(
        posteriorOnLatents=qK)
    qHSpikesTimes = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(
        posteriorOnLatents=qK)
    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink(
        preIntensityQuadTimes=qHQuadTimes,
        preIntensitySpikesTimes=qHSpikesTimes)
    klDiv = svGPFA.stats.klDivergence.KLDivergence(
        indPointsLocsKMS=indPointsLocsKMS)
    svlb = svGPFA.stats.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv)

    YStacked, _ = eLL._PointProcessELL__stackSpikeTimes(spikeTimes=YNonStacked)
    spikesTimesKMS.setTimes(times=YStacked)

#     qUParams0 = {"mean": qMu0, "cholVecs": srQSigma0Vecs}
#     kmsParams0 = {"kernels_params0": kernelsParams0,
#                   "inducing_points_locs0": Z0}
#     qKParams0 = {"posterior_on_ind_points": qUParams0,
#                  "kernels_matrices_store": kmsParams0}
#     qHParams0 = {"C0": C0, "d0": b0}
#     initial_params = {"posterior_on_latents": qKParams0,
#                      "embedding": qHParams0}
    eLLCalculationParams = {"leg_quad_points": legQuadPoints,
                            "leg_quad_weights": legQuadWeights}

#     svlb.setKernels(kernels=kernels)
#     svlb.setInitialParams(initial_params=initial_params)
    svlb.setMeasurements(measurements=YNonStacked)
    svlb.setELLCalculationParams(eLLCalculationParams=eLLCalculationParams)
    # svlb.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
#     svlb.setPriorCovRegParam(priorCovRegParam=1e-5) # Fix: need to read indPointsLocsKMSRegEpsilon from Matlab's CI test data
#     svlb.buildKernelsMatrices()

    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    spikesTimesKMS.setTimes(times=YStacked)
    Ktz_quad, KttDiag_quad = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)
    Ktz_spike, KttDiag_spike = spikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)
    kernels_matrices = dict(Kzz=Kzz, Kzz_inv=Kzz_inv,
                            Ktz_quad=Ktz_quad, KttDiag_quad=KttDiag_quad, 
                            Ktz_spike=Ktz_spike, KttDiag_spike=KttDiag_spike)
    lbEval = svlb.eval(variational_mean=qMu0, variational_cov=qSigma0, C=C0,
                       d=b0, kernels_matrices=kernels_matrices)

    assert(abs(lbEval+obj)<tol)

# def test_eval_poisson():
#     tol = 1e-5
#     dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_svGPFA.mat")
# 
#     mat = loadmat(dataFilename)
#     nLatents = len(mat['q_mu'])
#     nTrials = mat['q_mu'][0,0].shape[2]
#     qMu = [jax.device_put(mat['q_mu'][(i,0)]).astype("float64").transpose(2,0,1) for i in range(nLatents)]
#     qSVec = [jax.device_put(mat['q_sqrt'][(i,0)]).astype("float64").transpose(2,0,1) for i in range(nLatents)]
#     qSDiag = [jax.device_put(mat['q_diag'][(i,0)]).astype("float64").transpose(2,0,1) for i in range(nLatents)]
#     t_tmp = jax.device_put(mat['tt']).astype("float64").squeeze()
#     Z = [jax.device_put(mat['Z'][(i,0)]).astype("float64").transpose(2,0,1) for i in range(nLatents)]
#     Y = jax.device_put(mat['Y']).astype("float64").transpose(2,0,1)
#     C = jax.device_put(mat["C"]).astype("float64")
#     b = jax.device_put(mat["b"]).astype("float64").squeeze()
#     hermQuadPoints = jax.device_put(mat['xxHerm']).astype("float64")
#     hermQuadWeights = jax.device_put(mat['wwHerm']).astype("float64")
#     binWidth = jax.device_put(mat['BinWidth'])
#     obj = mat['obj'][0,0]
# 
#     # t_tmp \in nQuad and we want t \in nTrials x nQuad x 1
#     t = torch.ger(input=torch.ones(nTrials, .astype=torch.double), vec2=t_tmp).unsqueeze(dim=2)
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
#     qU = InducingPointsPrior(qMu=qMu, qSVec=qSVec, qSDiag=qSDiag, varRnk=torch.ones(3,.astype=torch.uint8))
#     kernelsMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=t, Y=Y)
#     qH = ApproxPosteriorForHForAllNeuronsQuadTimes(C=C, d=b, inducingPointsPrior=qU, kernelsMatricesStore=kernelsMatricesStore)
# 
#     eLL = PoissonExpectedLogLikelihood(approxPosteriorForHForAllNeuronsQuadTimes=qH, hermQuadPoints=hermQuadPoints, hermQuadWeights=hermQuadWeights, linkFunction=linkFunction, Y=Y, binWidth=binWidth)
#     klDiv = KLDivergence(kernelsMatricesStore=kernelsMatricesStore, inducingPointsPrior=qU)
#     svlb = SparseVariationalLowerBound(eLL=eLL, klDiv=klDiv)
#     lbEval = svlb.eval()
# 
#     assert(abs(lbEval+obj)<tol)
# 
if __name__=='__main__':
    test_eval_pointProcess()
    # test_eval_poisson()
