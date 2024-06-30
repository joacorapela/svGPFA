
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
    nLatents = mat['q_sqrt'].shape[0]
    qMu0list = [jax.device_put(mat['q_mu'][(i, 0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt'][(i, 0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag'][(i, 0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]

    qSigma0list = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0, diags=qSDiag0)

    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    matKLDiv = mat['KLd']
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel
            kernels_params0[k] = jnp.array([float(hprs[k,0][0].item()),
                                            float(hprs[k,0][1].item())])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel
            kernels_params0[k] = jnp.array([float(hprs[k,0][0].item())])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qMu0 = jnp.empty((len(qMu0list), qMu0list[0].shape[0],
                      qMu0list[0].shape[1]), dtype=jnp.double)
    qSigma0 = jnp.empty((len(qSigma0list), qSigma0list[0].shape[0],
                         qSigma0list[0].shape[1], qSigma0list[0].shape[2]),
                        dtype=jnp.double)
    for k in range(len(qMu0list)):
        qMu0 = qMu0.at[k, :, :].set(qMu0list[k][:, :, 0])
        qSigma0 = qSigma0.at[k, :, :, :].set(qSigma0list[k][:, :, :])

    Z0array = jnp.empty((len(Z0), Z0[0].shape[0], Z0[0].shape[1],
                         Z0[0].shape[2]), dtype=jnp.double)
    for k in range(len(Z0)):
        Z0array = Z0array.at[k, :, :, :].set(Z0[k])

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array,
        reg_param=reg_param)

    klDiv = svGPFA.stats.klDivergence.KLDivergence
    klDivEval = klDiv.evalSumAcrossLatentsAndTrials(
        vMean=qMu0, vCov=qSigma0, Kzz=Kzz, Kzz_cho=Kzz_cho)

    klError = abs(matKLDiv-klDivEval)

    assert(klError<tol)

if __name__=="__main__":
    test_evalSumAcrossLatentsTrials()
