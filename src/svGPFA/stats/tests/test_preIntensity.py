
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

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(kernels=kernels)
    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    quadTimesKMS.setTimes(times=t)
    Ktz, KttDiag = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qH = svGPFA.stats.preIntensity.LinearPreIntensityQuadTimes(
        posteriorOnLatents=qK)
    qHMu, qHVar = qH.computeMeansAndVars(variational_mean=qMu0,
                                         variational_cov=qSigma0, C=C0, d=b0,
                                         Kzz=Kzz, Kzz_inv=Kzz_inv,
                                         Ktz=Ktz, KttDiag=KttDiag)

    n_trials = len(qHMu)
    for r in range(n_trials):
        qHMuError = math.sqrt(((mu_h[r,:,:]-qHMu[k])**2).mean())
        assert(qHMuError<tol)
        qHVarError = math.sqrt(((var_h[r,:,:]-qHVar[k])**2).mean())
        assert(qHVarError<tol)

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
    t = jax.device_put(mat["ttQuad"].astype("float64").transpose(2, 0, 1))
    Z0 = [jax.device_put(mat["Z"][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Y = [jax.device_put(mat["Y"][tr,0].astype("float64")) for tr in range(nTrials)]
    C0 = jax.device_put(mat["C"].astype("float64"))
    b0 = jax.device_put(mat["b"].astype("float64"))
    mu_h = [jax.device_put(mat["mu_h_Spikes"][0,i].astype("float64").squeeze()) for i in range(nTrials)]
    var_h = [jax.device_put(mat["var_h_Spikes"][0,i].astype("float64").squeeze()) for i in range(nTrials)]
    index = [jax.device_put(mat["index"][i,0][:,0].astype("uint8"))-1 for i in range(nTrials)]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    qSigma0 = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
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

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(kernels=kernels)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(kernels=kernels)
    Kzz, Kzz_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0, reg_param=reg_param)
    spikesTimesKMS.setTimes(times=Y)
    Ktz, KttDiag = spikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    qK = svGPFA.stats.posteriorOnLatents.PosteriorOnLatents()
    qH = svGPFA.stats.preIntensity.LinearPreIntensitySpikesTimes(
        posteriorOnLatents=qK)
    qH.setNeuronForSpikeIndex(neuronForSpikeIndex=index)
    qHMu, qHVar = qH.computeMeansAndVars(variational_mean=qMu0,
                                         variational_cov=qSigma0, C=C0, d=b0,
                                         Kzz=Kzz, Kzz_inv=Kzz_inv,
                                         Ktz=Ktz, KttDiag=KttDiag)

    for i in range(len(mu_h)):
        qHMuError = math.sqrt(jnp.sum((mu_h[i]-qHMu[i])**2))/mu_h[i].shape[0]
        assert(qHMuError<tol)
        qHVarError = math.sqrt(jnp.sum((var_h[i]-qHVar[i])**2))/\
                     var_h[i].shape[0]
        assert(qHVarError<tol)

if __name__=="__main__":
    test_computeMeansAndVars_quadTimes()
    test_computeMeansAndVars_spikesTimes()
