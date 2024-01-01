
import sys
import os
import math
from scipy.io import loadmat
import numpy as np
import jax
import jax.numpy as jnp
import svGPFA.utils.miscUtils
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.variationalDist
import svGPFA.stats.posteriorOnLatents
import svGPFA.stats.preIntensity
import svGPFA.stats.expectedLogLikelihood

jax.config.update("jax_enable_x64", True)

def test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink():
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
    hermQuadPoints = jax.device_put(mat['xxHerm'].astype("float64"))
    hermQuadWeights = jax.device_put(mat['wwHerm'].astype("float64"))
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2, 0, 1))
    Elik = jax.device_put(mat['Elik'])
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

    YStacked, neuronForSpikeIndex = svGPFA.utils.miscUtils.stackSpikeTimes(spikeTimes=YNonStacked)

    linkFunction = jnp.exp

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    eLLCalculationParams = {"leg_quad_points": legQuadPoints,
                            "leg_quad_weights": legQuadWeights}

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

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=legQuadPoints)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=YStacked)
    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    Ktz_quad, KttDiag_quad = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)
    Ktz_spike, KttDiag_spike = spikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)
    kernels_matrices = dict(Kzz=Kzz, Kzz_inv=Kzz_inv,
                            Ktz_quad=Ktz_quad, KttDiag_quad=KttDiag_quad, 
                            Ktz_spike=Ktz_spike, KttDiag_spike=KttDiag_spike)

    sELL = eLL.evalSumAcrossTrialsAndNeurons(variational_mean=qMu0,
                                             variational_cov=qSigma0,
                                             C=C0, d=b0,
                                             kernels_matrices=kernels_matrices)

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

if __name__=="__main__":
    test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink()
    # test_evalSumAcrossTrialsAndNeurons_pointProcessQuad()
