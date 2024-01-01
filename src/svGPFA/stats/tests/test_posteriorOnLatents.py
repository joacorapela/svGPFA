
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
    tol = 5e-6
    reg_param = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [jax.device_put(mat["q_mu"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat["q_sqrt"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat["q_diag"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    t = jax.device_put(mat["ttQuad"].astype("float64").transpose(2, 0, 1))
    Z0 = [jax.device_put(mat["Z"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    mu_k = jax.device_put(mat["mu_k_Quad"].astype("float64").transpose(2,0,1))
    var_k = jax.device_put(mat["var_k_Quad"].astype("float64").transpose(2,0,1))
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    qSigma = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                          diags=qSDiag0)
    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = hprs[k,0][0].item() 
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = hprs[k,0][0].item()
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=t)
    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()

    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    Ktz, KttDiag = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    qKMu, qKVar = qK.computeMeansAndVars(variational_mean=qMu0,
                                         variational_cov=qSigma,
                                         Kzz=Kzz, Kzz_inv=Kzz_inv,
                                         Ktz=Ktz, KttDiag=KttDiag)

    for r in range(len(qKMu)):
        qKMuError = math.sqrt(((mu_k[r,:,:] - qKMu[r])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[r,:,:]-qKVar[r])**2).mean())
        assert(qKVarError<tol)

def test_computeMeansAndVars_spikesTimes():
    tol = 5e-6
    reg_param = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat["Z"].shape[0]
    nTrials = mat["Z"][0,0].shape[2]
    qMu0 = [jax.device_put(mat["q_mu"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat["q_sqrt"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat["q_diag"][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Z0 = [jax.device_put(mat["Z"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Y = [jax.device_put(mat["Y"][tr,0].astype("float64")) for tr in range(nTrials)]
    mu_k = [jax.device_put(mat["mu_k_Spikes"][0,tr].astype("float64")) for tr in range(nTrials)]
    var_k = [jax.device_put(mat["var_k_Spikes"][0,tr].astype("float64")) for tr in range(nTrials)]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    qSigma = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                          diags=qSDiag0)
    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = hprs[k,0][0].item() 
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = hprs[k,0][0].item()
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(kernels=kernels)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=Y)
    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()

    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    Ktz, KttDiag = spikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    qKMu, qKVar = qK.computeMeansAndVars(variational_mean=qMu0,
                                         variational_cov=qSigma,
                                         Kzz=Kzz, Kzz_inv=Kzz_inv,
                                         Ktz=Ktz, KttDiag=KttDiag)

    for tr in range(nTrials):
        qKMuError = math.sqrt(((mu_k[tr]-qKMu[tr])**2).mean())
        assert(qKMuError<tol)
        qKVarError = math.sqrt(((var_k[tr]-qKVar[tr])**2).mean())
        assert(qKVarError<tol)

if __name__=="__main__":
    test_computeMeansAndVars_quadTimes()
    test_computeMeansAndVars_spikesTimes()
