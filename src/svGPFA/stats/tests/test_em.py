
import sys
import io
import os
import math
import scipy.io
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
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = scipy.io.loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0list = [jax.device_put(mat['q_mu'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    C0 = jax.device_put(mat["C"].astype("float64"))
    b0 = jax.device_put(mat["b"].astype("float64"))
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2,0,1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2,0,1))
    obj = mat['obj'][0,0]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    YNonStacked_tmp = mat['YNonStacked']

    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = jax.device_put(YNonStacked_tmp[r,0][n,0][:,0]).astype("float64")

    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=YNonStacked)

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0,k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel
            lengthscale = float(hprs[k,0][0].item())
            period = hprs[k,0][1].item()
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qMu0 = jnp.empty((len(qMu0list), qMu0list[0].shape[0],
                      qMu0list[0].shape[1]), dtype=jnp.double)
    qSigma0list = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
    qSigma0 = jnp.empty((len(qSigma0list), qSigma0list[0].shape[0],
                         qSigma0list[0].shape[1], qSigma0list[0].shape[2]),
                        dtype=jnp.double)
    for k in range(len(qMu0list)):
        qMu0 = qMu0.at[k, :, :].set(qMu0list[k][:, :, 0])
        qSigma0 = qSigma0.at[k, :, :, :].set(qSigma0list[k][:, :, :])
    qSigma0_chol_vecs = svGPFA.utils.miscUtils.getCholVecsFromCov(cov=qSigma0)

    Z0array = jnp.empty((len(Z0), Z0[0].shape[0], Z0[0].shape[1],
                         Z0[0].shape[2]), dtype=jnp.double)
    for k in range(len(Z0)):
        Z0array = Z0array.at[k, :, :, :].set(Z0[k])

    params0 = dict(
        variational_mean = qMu0,
        variational_chol_vecs = qSigma0_chol_vecs,
        C = C0,
        d = b0,
        kernels_params = kernels_params0,
        ind_points_locs = Z0array,
    )
    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights,
            reg_param=reg_param)

    lbEval = em._eval_func_params_as_list(params=params0)
    assert(abs(lbEval-obj)<tol)

def test_maximize_pointProcess_JAX(reg_param=1e-5, maxiter=400,
                                   max_stepsize=200.0, tol = 1e-5, jit=True,
                                   verbose=True):
    dataFilename = os.path.join(os.path.dirname(__file__), "data/variationalEM.mat")

    mat = scipy.io.loadmat(dataFilename)
    nLatents = len(mat['Z0'])
    nTrials = mat['Z0'][0,0].shape[2]
    qMu0list = [jax.device_put(mat['q_mu0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag0'][(0,i)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    # srQSigma0Vecs = svGPFA.utils.miscUtils.getSRQSigmaVec(qSVec=qSVec0, qSDiag=qSDiag0)
    Z0 = [jax.device_put(mat['Z0'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    C0 = jax.device_put(mat["C0"].astype("float64"))
    b0 = jax.device_put(mat["b0"].astype("float64"))
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2, 0, 1))
    YNonStacked_tmp = mat['YNonStacked']

    nNeurons = YNonStacked_tmp[0,0].shape[0]
    YNonStacked = [[[] for n in range(nNeurons)] for r in range(nTrials)]
    for r in range(nTrials):
        for n in range(nNeurons):
            YNonStacked[r][n] = jax.device_put(YNonStacked_tmp[r,0][n,0][:,0].astype("float64"))

    spikes_times_array, valid_spikes_times_mask = \
        svGPFA.utils.miscUtils.buildSpikesTimesArray(spikes_times=YNonStacked)

    kernelNames = mat["kernelNames"]
    hprs = mat["hprs0"]
    leasLowerBound = mat['lowerBound'][0,0]

    qMu0 = jnp.empty((len(qMu0list), qMu0list[0].shape[0],
                      qMu0list[0].shape[1]), dtype=jnp.double)
    qSigma0list = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
    qSigma0 = jnp.empty((len(qSigma0list), qSigma0list[0].shape[0],
                         qSigma0list[0].shape[1], qSigma0list[0].shape[2]),
                        dtype=jnp.double)
    for k in range(len(qMu0list)):
        qMu0 = qMu0.at[k, :, :].set(qMu0list[k][:, :, 0])
        qSigma0 = qSigma0.at[k, :, :, :].set(qSigma0list[k][:, :, :])
    qSigma0_chol_vecs = svGPFA.utils.miscUtils.getCholVecsFromCov(cov=qSigma0)

    Z0array = jnp.empty((len(Z0), Z0[0].shape[0], Z0[0].shape[1],
                         Z0[0].shape[2]), dtype=jnp.double)
    for k in range(len(Z0)):
        Z0array = Z0array.at[k, :, :, :].set(Z0[k])

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

    em = svGPFA.stats.em.EM_JAXopt
    em.init(spikesTimesArray=spikes_times_array,
            validSpikesTimesMask=valid_spikes_times_mask, kernels=kernels,
            legQuadPoints=legQuadPoints, legQuadWeights=legQuadWeights,
            reg_param=reg_param)

    params0 = dict(
        variational_mean = qMu0,
        variational_chol_vecs = qSigma0_chol_vecs,
        C = C0,
        d = b0,
        kernels_params = kernels_params0,
        ind_points_locs = Z0array,
    )
    optim_params = dict(
        maxiter=maxiter,
        tol=1e-6,
        max_stepsize=5.0,
        jit=True,
        # verbose=verbose,
    )

    res = em.maximize(params0=params0, optim_params=optim_params)
    # res = em.maximizeInSteps(params0=params0, optim_params=optim_params)
    lower_bound = -res.state.value
    assert(lower_bound>leasLowerBound)

if __name__=='__main__':
    test_emJAX__eval_func()
    test_maximize_pointProcess_JAX(reg_param = 1e-5, maxiter=400,
                                   max_stepsize=200.0, tol = 1e-5, jit=True,
                                   verbose=False)

