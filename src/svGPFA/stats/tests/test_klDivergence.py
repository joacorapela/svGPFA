
import sys
import os
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import svGPFA.utils.miscUtils
import svGPFA.stats.variationalDist
import svGPFA.stats.kernelsMatricesStore
import svGPFA.stats.klDivergence

jax.config.update("jax_enable_x64", True)

def test_evalSumAcrossLatentsTrials():
    tol = 1e-5
    reg_param = 1e-5 # Fix: need to read indPointsLocsKMSEpsilon from Matlab's CI test data
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['q_sqrt'].shape[1]
    variational_mean = [jax.device_put(mat['q_mu'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    variational_cov = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(
        vecs=qSVec0, diags=qSDiag0)

    ind_points_locs0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    matKLDiv = mat['KLd']
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            kernels_params0[k] = {"scale": 1.0,
                                  "lengthscale": hprs[k,0][0].item(),
                                  "lengthscaleScale": 1.0,
                                  "period": hprs[k,0][1].item(),
                                  "periodScale": 1.0,
                                 }
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            kernels_params0[k] = {"scale": 1.0,
                                  "lengthscale": hprs[k,0][0].item(),
                                  "lengthscaleScale": 1.0,
                                 }
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    prior_cov, prior_cov_inv = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=ind_points_locs0,
        reg_param=reg_param)

    klDiv = svGPFA.stats.klDivergence.KLDivergence(
        indPointsLocsKMS=indPointsLocsKMS)
    klDivEval = klDiv.evalSumAcrossLatentsAndTrials(
        variational_mean=variational_mean,
        variational_cov=variational_cov,
        prior_cov=prior_cov,
        prior_cov_inv=prior_cov_inv,
    )

    klError = abs(matKLDiv-klDivEval)

    assert(klError<tol)

if __name__=="__main__":
    test_evalSumAcrossLatentsTrials()
