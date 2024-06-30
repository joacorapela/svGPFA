
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

jax.config.update("jax_enable_x64", True)

def test_computeMeansAndVars():
    tol = 1e-6
    reg_param = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0list = [jax.device_put(mat["q_mu"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat["q_sqrt"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat["q_diag"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    t = jax.device_put(mat["ttQuad"].astype("float64").transpose(2, 0, 1))
    Z0 = [jax.device_put(mat["Z"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Y = [jax.device_put(mat["Y"][tr,0].astype("float64")) for tr in range(nTrials)]
    C0 = jax.device_put(mat["C"].astype("float64"))
    b0 = jax.device_put(mat["b"].astype("float64"))
    mu_h = jax.device_put(mat["mu_h_Quad"].astype("float64").transpose(2,0,1))
    var_h = jax.device_put(mat["var_h_Quad"].astype("float64").transpose(2,0,1))
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    qSigma0 = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel
            lengthscale = float(hprs[k,0][0].item())
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qSigmaList = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                              diags=qSDiag0)
    qMu0 = jnp.empty((len(qMu0list), qMu0list[0].shape[0],
                      qMu0list[0].shape[1]), dtype=jnp.double)
    qSigma = jnp.empty((len(qSigmaList), qSigmaList[0].shape[0],
                        qSigmaList[0].shape[1], qSigmaList[0].shape[2]),
                       dtype=jnp.double)
    for k in range(len(qMu0list)):
        qMu0 = qMu0.at[k, :, :].set(qMu0list[k][:, :, 0])
        qSigma = qSigma.at[k, :, :, :].set(qSigmaList[k][:, :, :])

    Z0array = jnp.empty((len(Z0), Z0[0].shape[0], Z0[0].shape[1],
                         Z0[0].shape[2]), dtype=jnp.double)
    for k in range(len(Z0)):
        Z0array = Z0array.at[k, :, :, :].set(Z0[k])

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
    quadTimesKMS.init(kernels=kernels, t=t)
    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array, reg_param=reg_param)
    Ktz = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qH = svGPFA.stats.preIntensity.LinearPreIntensity
    qHMu = qH.computeMeans(vMean=qMu0, C=C0, d=b0, Kzz_cho=Kzz_cho, Ktz=Ktz)
    qHVar = qH.computeVars(vCov=qSigma, C=C0, Kzz=Kzz, Kzz_cho=Kzz_cho, Ktz=Ktz,
                           KttDiag=1.0)

    n_trials = len(qHMu)
    for r in range(n_trials):
        qHMuError = math.sqrt(((mu_h[r,:,:].T-qHMu[r,:,:])**2).mean())
        assert(qHMuError<tol)
        qHVarError = math.sqrt(((var_h[r,:,:].T-qHVar[r,:,:])**2).mean())
        assert(qHVarError<tol)


if __name__=="__main__":
    test_computeMeansAndVars()
