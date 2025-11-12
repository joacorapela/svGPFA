
import os
import pickle
import scipy.io
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

def test_spike_times_array_and_mask():
    dataFilename = os.path.join(os.path.dirname(__file__),
                                "data/spike_times_data.pickle")
    with open(dataFilename, "rb") as f:
        load_res = pickle.load(f)
    # spikes_times[r][n]: list of spikes times for trial r and neuron n
    spikes_times = load_res["spikes_times"]
    # spikes_times_array: float64 array \in n_trials x n_spikes x 1
    spikes_times_array = load_res["spikes_times_array"]
    # valid_spikes_times_mask: bollean array \in n_trials x n_neurons x n_spikes
    valid_spikes_times_mask = load_res["valid_spikes_times_mask"]

    n_trials = valid_spikes_times_mask.shape[0]
    n_neurons = valid_spikes_times_mask.shape[1]
    n_spikes = valid_spikes_times_mask.shape[2]

    recovered_spikes_times = []
    for r in range(n_trials):
        recovered_spikes_times_r = []
        for n in range(n_neurons):
            recovered_spikes_times_rn = \
                spikes_times_array[r, valid_spikes_times_mask[r, n, :], 0]
            recovered_spikes_times_r.append(recovered_spikes_times_rn)
        recovered_spikes_times.append(recovered_spikes_times_r)

    for r in range(n_trials):
        for n in range(n_neurons):
            assert(set(spikes_times[r][n].tolist()) == set(recovered_spikes_times[r][n].tolist()))


def test_jnp_where(n_trials=2, tol=1e-6):
    dataFilename = os.path.join(os.path.dirname(__file__),
                                "data/spike_times_data.pickle")
    with open(dataFilename, "rb") as f:
        load_res = pickle.load(f)
    # valid_spikes_times_mask: bollean array \in n_trials x n_neurons x n_spikes
    valid_spikes_times_mask = load_res["valid_spikes_times_mask"]
    valid_spikes_times_mask = valid_spikes_times_mask[:n_trials, :, :]

    n_trials = valid_spikes_times_mask.shape[0]
    n_neurons = valid_spikes_times_mask.shape[1]
    n_spikes = valid_spikes_times_mask.shape[2]

    # qHMuSpikes \in n_trials x n_neurons x n_spikes
    qHMuSpikes = np.random.normal(size=(n_trials, n_neurons, n_spikes))
    eLogLinkValues = jnp.where(valid_spikes_times_mask, qHMuSpikes, 0.0)
    sELLTerm2 = jnp.sum(eLogLinkValues)

    sELLTerm2_new = 0.0
    for r in range(n_trials):
        print(f"*** Processing trial {r}: sELLTerm2={sELLTerm2}, sELLTerm2_new={sELLTerm2_new}")
        for n in range(n_neurons):
            print(f"Processing neuron {n}: sELLTerm2={sELLTerm2}, sELLTerm2_new={sELLTerm2_new}")
            for i in range(n_spikes):
                if valid_spikes_times_mask[r, n, i]:
                    sELLTerm2_new += qHMuSpikes[r, n, i]

    print(f"Processing done: sELLTerm2={sELLTerm2}, sELLTerm2_new={sELLTerm2_new}")
    assert(abs(sELLTerm2 - sELLTerm2_new) < tol)

def test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink():
    tol = 3e-4
    reg_param = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Estep_Objective_PointProcess_svGPFA.mat")

    mat = scipy.io.loadmat(dataFilename)
    nLatents = len(mat['Z'])
    nTrials = mat['Z'][0,0].shape[2]
    qMu0list = [jax.device_put(mat['q_mu'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSVec0 = [jax.device_put(mat['q_sqrt'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    qSDiag0 = [jax.device_put(mat['q_diag'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    t = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    C0 = jax.device_put(mat["C"].astype("float64"))
    # b0 = jax.device_put(mat["b"].astype("float64").squeeze())
    b0 = jax.device_put(mat["b"].astype("float64"))
    hermQuadPoints = jax.device_put(mat['xxHerm'].astype("float64"))
    hermQuadWeights = jax.device_put(mat['wwHerm'].astype("float64"))
    legQuadPoints = jax.device_put(mat['ttQuad'].astype("float64").transpose(2, 0, 1))
    legQuadWeights = jax.device_put(mat['wwQuad'].astype("float64").transpose(2, 0, 1))
    Elik = jax.device_put(mat['Elik'])
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    YNonStacked_tmp = mat['YNonStacked']

    qSigma0List = svGPFA.utils.miscUtils.buildRank1PlusDiagCov(vecs=qSVec0,
                                                           diags=qSDiag0)
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
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    qMu0 = jnp.empty((len(qMu0list), qMu0list[0].shape[0],
                      qMu0list[0].shape[1]), dtype=jnp.double)
    qSigma0 = jnp.empty((len(qSigma0List), qSigma0List[0].shape[0],
                         qSigma0List[0].shape[1], qSigma0List[0].shape[2]),
                        dtype=jnp.double)
    for k in range(len(qMu0list)):
        qMu0 = qMu0.at[k, :, :].set(qMu0list[k][:, :, 0])
        qSigma0 = qSigma0.at[k, :, :, :].set(qSigma0List[k][:, :, :])

    Z0array = jnp.empty((len(Z0), Z0[0].shape[0], Z0[0].shape[1],
                         Z0[0].shape[2]), dtype=jnp.double)
    for k in range(len(Z0)):
        Z0array = Z0array.at[k, :, :, :].set(Z0[k])

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol
    indPointsLocsKMS.init(kernels=kernels)
    quadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndQuadTimesKMS
    quadTimesKMS.init(kernels=kernels, t=legQuadPoints)
    spikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndSpikesTimesKMS
    spikesTimesKMS.init(kernels=kernels, t=spikes_times_array)
    Kzz, Kzz_cho = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array, reg_param=reg_param)
    Ktz_quad = quadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array)
    Ktz_spikes = spikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0array)

    eLL = svGPFA.stats.expectedLogLikelihood.PointProcessELLExpLink
    eLL.init(legQuadWeights=legQuadWeights,
             validSpikesTimesMask=valid_spikes_times_mask)
    sELL = eLL.evalSumAcrossTrialsAndNeurons(vMean=qMu0, vCov=qSigma0,
                                             C=C0, d=b0,
                                             Kzz=Kzz, Kzz_cho=Kzz_cho,
                                             KtzQuad=Ktz_quad,
                                             KtzSpikes=Ktz_spikes,
                                             KttDiag=1.0)

    sELLerror = abs(sELL-Elik)

    assert(sELLerror<tol)

if __name__=="__main__":
    # test_evalSumAcrossTrialsAndNeurons_pointProcessExpLink()
    # test_spike_times_array_and_mask()
    test_jnp_where()
