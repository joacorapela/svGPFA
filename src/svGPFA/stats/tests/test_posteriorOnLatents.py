
import sys
import pdb
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

jax.config.update("jax_enable_x64", True)

def test_computeMeansAndVars_quadTimes():
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
    mu_k = jax.device_put(mat["mu_k_Quad"].astype("float64").transpose(2,0,1))
    var_k = jax.device_put(mat["var_k_Quad"].astype("float64").transpose(2,0,1))
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

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

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel
            lengthscale = hprs[k,0][0].item()
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel
            lengthscale = hprs[k,0][0].item()
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
    quadTimesKMS.init(kernels=kernels, t=t)
    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents

    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array, reg_param=reg_param)
    Ktz = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array)

    qKMu = qK.computeMeans(vMean=qMu0, Kzz_cho=Kzz_cho, Ktz=Ktz)
    qKVar = qK.computeVars(vCov=qSigma, Kzz=Kzz, Kzz_cho=Kzz_cho,
                           Ktz=Ktz, KttDiag=1.0)

    for r in range(qKMu.shape[1]):
        qKMuError = math.sqrt(((mu_k[r,:,:].T - qKMu[:, r, :])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[r,:,:].T - qKVar[:, r, :])**2).mean())
        assert(qKVarError<tol)


if __name__=="__main__":
    test_computeMeansAndVars_quadTimes()
    # test_computeMeansAndVars_spikesTimes()
