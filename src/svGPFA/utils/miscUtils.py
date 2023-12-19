import scipy.io
import math
import pandas as pd
import scipy
import sklearn.metrics
import jax
import jax.numpy as jnp
import warnings

# from . import my_globals
import svGPFA.stats.kernels
import gcnu_common.numerical_methods.utils
import gcnu_common.stats.gaussianProcesses.eval


def separateNeuronsSpikeTimesByTrials(neurons_spike_times, epochs_times,
                                      trials_start_times_rel,
                                      trials_end_times_rel):
    n_trials = len(epochs_times)
    n_neurons = len(neurons_spike_times)
    trials_spikes_times = [[] for r in range(n_trials)]
    for r in range(n_trials):
        trial_epoch_time = epochs_times[r]
        trial_start_time_rel = trials_start_times_rel[r]
        trial_end_time_rel = trials_end_times_rel[r]
        trial_spikes_times = [[] for n in range(n_neurons)]
        for n in range(n_neurons):
            neuron_spikes_times_rel = neurons_spike_times[n]-trial_epoch_time
            trial_neuron_spikes_times = neuron_spikes_times_rel[
                jnp.logical_and(trial_start_time_rel <= neuron_spikes_times_rel,
                               neuron_spikes_times_rel < trial_end_time_rel)]
            trial_spikes_times[n] = trial_neuron_spikes_times
        trials_spikes_times[r] = trial_spikes_times
    return trials_spikes_times


def buildKernels(kernels_types, kernels_params):
    n_latents = len(kernels_types)
    kernels = [None for k in range(n_latents)]

    for k, kernel_type in enumerate(kernels_types):
        if kernels_types[k] == "exponentialQuadratic":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
        elif kernels_types[k] == "periodic":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
        else:
            raise ValueError(f"Invalid kernels type: {kernels_types[k]}")
        kernels[k].setParams(kernels_params[k])
    return kernels


def orthonormalizeLatentsMeans(latents_means, C):
    U, S, Vh = jnp.linalg.svd(C)
    orthoMatrix = Vh.T*S
    n_trials = len(latents_means)
    oLatentsMeans = [[] for r in range(n_trials)]
    for r in range(n_trials):
        oLatentsMeans[r] = jnp.matmul(latents_means[r], orthoMatrix)
    return oLatentsMeans


def getPropSamplesCovered(sample, mean, std, percent=.95):
    if percent==.95:
        factor = 1.96
    else:
        raise ValueError("percent=0.95 is the only option implemented at the moment")
    covered = jnp.logical_and(mean-factor*std<=sample, sample<mean+factor*std)
    coverage = jnp.count_nonzero(covered)/float(len(covered))
    return coverage

def getCIFs(C, d, latents):
    n_trials = latents.shape[0]
    nLatents = latents.shape[2]
    nSamples = latents.shape[1]
    nNeurons = C.shape[0]
    embeddings = jnp.empty((n_trials, nSamples, nNeurons))
    for r in range(n_trials):
        embeddings[r,:,:] = jnp.matmul(latents[r,:,:], jnp.transpose(C, 0, 1))+d[:,0]
    CIFs = jnp.exp(embeddings)
    return CIFs


def computeSpikeRates(trials_times, spikes_times):
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    spikes_rates = jnp.empty((n_trials, n_neurons))
    for r in range(n_trials):
        trial_duration = jnp.max(trials_times[r])-jnp.min(trials_times[r])
        for n in range(n_neurons):
            spikes_rates[r, n] = len(spikes_times[r][n])/trial_duration
    return spikes_rates


def saveDataForMatlabEstimations(qMu, qSVec, qSDiag, C, d,
                                 indPointsLocs,
                                 legQuadPoints, legQuadWeights,
                                 kernelsTypes, kernelsParams,
                                 spikesTimes,
                                 indPointsLocsKMSRegEpsilon, trialsLengths,
                                 latentsTrialsTimes,
                                 emMaxIter, eStepMaxIter, mStepEmbeddingMaxIter,
                                 mStepKernelsMaxIter, mStepIndPointsMaxIter,
                                 saveFilename):
    n_trials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    nLatents = len(qMu)
    # indPointsLocsKMSEpsilon = jnp.array(indPointsLocsKMSEpsilon)
    mdict = dict(n_trials=n_trials, nNeurons=nNeurons, nLatents=nLatents,
                 C=C.numpy(), d=jnp.reshape(input=d, shape=(-1,1)).numpy(),
                 legQuadPoints=legQuadPoints.numpy(),
                 legQuadWeights=legQuadWeights.numpy(),
                 indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon,
                 trialsLengths=trialsLengths.numpy(),
                 emMaxIter=emMaxIter, eStepMaxIter=eStepMaxIter,
                 mStepEmbeddingMaxIter=mStepEmbeddingMaxIter,
                 mStepKernelsMaxIter=mStepKernelsMaxIter,
                 mStepIndPointsMaxIter=mStepIndPointsMaxIter)
    for k in range(nLatents):
        mdict.update({"kernelType_{:d}".format(k): kernelsTypes[k]})
        mdict.update({"qMu_{:d}".format(k): qMu[k].numpy().astype(jnp.float64)})
        mdict.update({"qSVec_{:d}".format(k): qSVec[k].numpy().astype(jnp.float64)})
        mdict.update({"qSDiag_{:d}".format(k): qSDiag[k].numpy().astype(jnp.float64)})
        mdict.update({"kernelsParams_{:d}".format(k):
                      kernelsParams[k].numpy().astype(jnp.float64)})
        mdict.update({"indPointsLocs_{:d}".format(k):
                      indPointsLocs[k].numpy().astype(jnp.float64)})
        mdict.update({"qMu_{:d}".format(k): qMu[k].numpy().astype(jnp.float64)})
        mdict.update({"qSVec_{:d}".format(k): qSVec[k].numpy().astype(jnp.float64)})
        mdict.update({"qSDiag_{:d}".format(k): qSDiag[k].numpy().astype(jnp.float64)})
        mdict.update({"latentsTrialsTimes_{:d}".format(k):
                      latentsTrialsTimes[k].numpy().astype(jnp.float64)})
    for r in range(n_trials):
        for n in range(nNeurons):
            mdict.update({"spikesTimes_{:d}_{:d}".format(r, n):
                          spikesTimes[r][n].numpy().astype(jnp.float64)})
    scipy.io.savemat(file_name=saveFilename, mdict=mdict)

def getCholFromVec(vec):
    """Build Cholesky lower-triangular matrix from its vector representation.

    :param vec: vector respresentation of the lower-triangular Cholesky factor
    :type  vector: :class:`jnp.Tensor`
    :param nIndPoints: number of inucing opoints.
    :type  nIndPoints: int
    """
    Pk = len(vec)
    nindPoints = int((math.sqrt(1 + 8 * Pk) - 1) / 2)
    chol = jnp.zeros((nIndPoints, nIndPoints), dtype=jnp.double)
    tril_indices = jnp.tril_indices(nIndPoints)
    chol[tril_indices[0],tril_indices[1]] = vec
    return chol

def buildCovsFromCholVecs(chol_vecs):
    """Build covariances from vector respresntations of their Cholesky
    descompositions.

    :param chol_vecs: vector respresentations of the lower-triangular Cholesky factors
    :type  list of length n_latents shuch that chol_vecs[k] \in (n_trials, Pk, 1) where Pk=(n_ind_points[k] * (n_ind_points[k] + 1)) / 2
    """
    # Pk = (n_ind_points * (n_ind_points + 1)) / 2 then
    # n_ind_points = (math.sqrt(1 + 8 * M) - 1) / 2
    n_latents = len(chol_vecs)
    n_trials = chol_vecs[0].shape[0]
    covs = [None] * n_latents
    for k in range(n_latents):
        Pk = chol_vecs[k].shape[1]
        n_ind_points_k = int((math.sqrt(1 + 8 * Pk) - 1) / 2)
        tril_indices_k = jnp.tril_indices(n_ind_points_k)
        chols_k = jnp.zeros((n_trials, n_ind_points_k, n_ind_points_k), dtype=jnp.double)
        for r in range(n_trials):
            chols_k = chols_k.at[r, tril_indices_k[0], tril_indices_k[1]].set(chol_vecs[k][r, :, 0])
        covs[k] = jnp.matmul(chols_k, jnp.transpose(chols_k, (0, 2, 1)))
    return covs

def getQSVecsAndQSDiagsFromQSCholVecs(qsCholVecs):
    # qsCholVecs[k] \in nTrial x Pk
    nLatents = len(qsCholVecs)
    n_trials = qsCholVecs[0].shape[0]
    qSVec = [[] for k in range(nLatents)]
    qSDiag = [[] for k in range(nLatents)]
    for k in range(nLatents):
        Pk = qsCholVecs[k].shape[1]
        nIndPointsK = int((-1.0+math.sqrt(1+8*Pk))/2.0)
        qSVec[k] = jnp.empty(n_trials, nIndPointsK, 1, dtype=jnp.double)
        qSDiag[k] = jnp.empty(n_trials, nIndPointsK, 1, dtype=jnp.double)
        for r in range(n_trials):
            qSRSigmaKR = getCholFromVec(vec=qsCholVecs[k][r, :, 0], nIndPoints=nIndPointsK)
            qSigmaKR = jnp.matmul(qSRSigmaKR, jnp.transpose(qSRSigmaKR, 0, 1))
            qSDiagKR = jnp.diag(qSigmaKR)
            qSigmaKR = qSigmaKR - jnp.diag(qSDiagKR)
            eValKR, eVecKR = jnp.eig(qSigmaKR, eigenvectors=True)
            maxEvalIKR = jnp.argmax(eValKR, dim=0)[0]
            qSVecKR = eVecKR[:, maxEvalIKR]*jnp.sqrt(eValKR[maxEvalIKR, 0])
            qSVec[k][r, :, 0] = qSVecKR
            qSDiag[k][r, :, 0] = qSDiagKR
    return qSVec, qSDiag


def clock(func):
    def clocked(*args,**kargs):
        t0 = time.perf_counter()
        result = func(*args,**kargs)
        elapsed = time.perf_counter()-t0
        name = func.__name__
        if len(args)>0:
            arg_str = ', '.join(repr(arg) for arg in args)
        else:
            arg_str = None
        if len(kargs)>0:
            keys = kargs.keys()
            values = kargs.values()
            karg_str = ', '.join(key + "=" + repr(value) for key in keys for value in values)
        else:
            karg_str = None
        if arg_str is not None and karg_str is not None:
            print('[%0.8fs] %s(%s,%s) -> %r' % (elapsed, name, arg_str, karg_str, result))
        elif arg_str is not None:
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, arg_str, result))
        elif karg_str is not None:
            print('[%0.8fs] %s(%s) -> %r' % (elapsed, name, karg_str, result))
        else:
            print('[%0.8fs] %s() -> %r' % (elapsed, name, result))
        return result
    return clocked

def chol3D(K):
#     if my_globals.raise_exception:
#         raise ValueError("Test error in chol3D")
    # Kchol = jnp.linalg.cholesky(K)
    Kchol, _ = jax.scipy.linalg.cho_factor(K, lower=True)
    return Kchol

chol3D_jitted = jax.jit(chol3D)

def pinv3D(K, rcond=1e-15):
    Kpinv = jnp.zeros(K.shape, dtype=K.dtype, device=K.device)
    nTrial = K.shape[0]
    for i in range(nTrial):
        Kpinv[i,:,:] = jnp.linalg.pinv(K[i,:,:], rcond=rcond)
    return Kpinv


def getLegQuadPointsAndWeights(n_quad, trials_start_times, trials_end_times):
    n_trials = len(trials_start_times)
    assert(n_trials == len(trials_end_times))
    leg_quad_points = [jnp.empty((n_quad[r], 1), dtype=dtype)
                       for r in range(n_trials)]
    leg_quad_weights = [jnp.empty((n_quad[r], 1), dtype=dtype)
                        for r in range(n_trials)]
    for r in range(n_trials):
        leg_quad_points[r][:, 0], leg_quad_weights[r][:, 0] = \
                gcnu_common.numerical_methods.utils.leggaussVarLimits(
                    n=n_quad[r], a=trials_start_times[r], b=trials_end_times[r])
    return leg_quad_points, leg_quad_weights


def getTrialsTimes(start_times, end_times, n_steps):
    assert(len(start_times) == len(end_times))
    n_trials = len(start_times)
    trials_times = jnp.empty((n_trials, n_steps, 1), dtype=jnp.double)
    for r in range(n_trials):
        trials_times[r, :, 0] = jnp.linspace(start_times[r], end_times[r],
                                               n_steps)
    return trials_times


def computeSpikeClassificationROC(spikes_times, cif_times, cif_values,
                                  highres_bin_size=1e-3):
    f = scipy.interpolate.interp1d(cif_times, cif_values)
    cif_times_highres = jnp.arange(cif_times[0], cif_times[-1],
                                  highres_bin_size)
    cif_values_highres = f(cif_times_highres)
    bins = pd.interval_range(start=cif_times[0].item(),
                             end=cif_times[-1].item(),
                             periods=len(cif_times_highres))
    cut_res = pd.cut(spikes_times, bins=bins, retbins=False)
    Y = cut_res.value_counts().values
    indicesMoreThanOneSpikes = (Y > 1).nonzero()
    if len(indicesMoreThanOneSpikes) > 0:
        warnings.warn("Found more than one spike in {:d} bins".format(
            len(indicesMoreThanOneSpikes)))
        Y[indicesMoreThanOneSpikes] = 1.0
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, cif_values_highres,
                                                     pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def getLatentsMeansAndSTDs(meansFuncs, kernels, trialsTimes):
    n_trials = len(trialsTimes)
    nLatents = len(kernels)
    latents_means = [[] for r in range(n_trials)]
    latents_STDs = [[] for r in range(n_trials)]

    for r in range(n_trials):
        latents_means[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        latents_STDs[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            gp = gcnu_common.stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[k], kernel=kernels[k])
            latents_means[r][k,:] = gp.mean(t=trialsTimes[r])
            latents_STDs[r][k,:] = gp.std(t=trialsTimes[r])
    return latents_means, latents_STDs

def getLatentsSTDs(kernels, trialsTimes):
    n_trials = len(trialsTimes)
    nLatents = len(kernels)
    latents_STDs = [[] for r in range(n_trials)]

    for r in range(n_trials):
        latents_STDs[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            latents_STDs[r][k,:] = kernels[k].buildKernelMatrixDiag(X=trialsTimes[r]).sqrt()
    return latents_STDs

# def getLatentsMeanFuncsSamples(latents_meansFuncs, trialsTimes, dtype):
#     n_trials = len(latents_meansFuncs)
#     nLatents = len(latents_meansFuncs[0])
#     latents_meansFuncsSamples = [[] for r in range(n_trials)]
#     for r in range(n_trials):
#         latents_meansFuncsSamples[r] = jnp.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
#         for k in range(nLatents):
#             latents_meansFuncsSamples[r][k,:] = latents_meansFuncs[r][k](t=trialsTimes[r])

def getLatentsSamplesMeansAndSTDsFromSampledMeans(n_trials, sampledMeans, kernels, trialsTimes, latentsGPRegularizationEpsilon):
    nLatents = len(kernels)
    latents_samples = [[] for r in range(n_trials)]
    latents_means = [[] for r in range(n_trials)]
    latents_STDs = [[] for r in range(n_trials)]

    for r in range(n_trials):
        latents_samples[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        latents_means[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        latents_STDs[r] = jnp.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            print("Procesing trial {:d} and latent {:d}".format(r+1, k+1))
            mean = sampledMeans[r,:,k]
            cov = kernels[k].buildKernelMatrix(trialsTimes[r])
            cov = cov + latentsGPRegularizationEpsilon*jnp.eye(cov.shape[0])
            std = jnp.diag(cov).sqrt()
            mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            sample = jnp.from_numpy(mn.rvs())
            latents_samples[r][k,:] = sample
            latents_means[r][k,:] = mean
            latents_STDs[r][k,:] = std
            # plt.plot(trialsTimes[r], mean, label="mean")
            # plt.plot(trialsTimes[r], sample, label="sample")
            # plt.xlabel("Time (sec)")
            # plt.ylabel("Value")
            # plt.title("Latent {:d}".format(k))
            # plt.legend()
            # plt.show()
    return latents_samples, latents_means, latents_STDs

def getDiagIndicesIn3DArray(N, M):
    frameDiagIndices = jnp.arange(N)*(N+1)
    frameStartIndices = jnp.arange(M)*N**2
    # jnp way of computing an outer sum
    diagIndices = (frameDiagIndices.reshape(-1,1)+frameStartIndices).flatten()
    answer = diagIndices.sort()
    return answer

def build3DdiagFromDiagVector(v, N, M):
    assert(len(v)==N*M)
    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    D = jnp.zeros(M*N*N)
    D = D.at[diagIndices].set(v)
    reshapedD = D.reshape((M, N, N))
    return reshapedD

def buildRank1PlusDiagCov(vecs, diags):
    nLatents = len(vecs)
    n_trials = vecs[0].shape[0]
    covs = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndK = diags[k].shape[1]
        # qq \in n_trials x nInd[k] x 1
        qq = vecs[k].reshape((n_trials, nIndK, 1))
        # dd \in n_trials x nInd[k] x 1
        nIndKVarRnkK = vecs[k].shape[1]
        dd = build3DdiagFromDiagVector(v=(diags[k].flatten())**2, M=n_trials, N=nIndKVarRnkK)
        # covs[k] \in n_trials x nInd[k] x nInd[k]
        covs[k] = jnp.matmul(qq, jnp.transpose(qq, (0, 2, 1))) + dd
    return covs

# def getCholVecs(cov):
#     nLatents = len(cov)
#     n_trials = cov[0].shape[0]
#     # qSigma = buildRank1PlusDiagCov(vecs=qSVec, diags=qSDiag)
#     cholVec = [[None] for k in range(nLatents)]
#     for k in range(nLatents):
#         nIndPointsK = cov[k].shape[1]
#         Pk = int((nIndPointsK+1)*nIndPointsK/2)
#         cholVec[k] = jnp.empty((n_trials, Pk, 1))
#         for r in range(n_trials):
#             cholKR = jnp.linalg.cholesky(cov[k][r,:,:])
#             tril_indices = jnp.tril_indices(nIndPointsK)
#             cholKRVec = cholKR[tril_indices[0], tril_indices[1]]
#             cholVec[k] = cholVec[k].at[r,:,0].set(cholKRVec)
#     return cholVec

# def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPointLoc):
#     nLatents = len(nIndPointsPerLatent)
#     n_trials = len(trialsLengths)
#
#     Z0 = [[] for k in range(nLatents)]
#     for k in range(nLatents):
#         Z0[k] = jnp.empty((n_trials, nIndPointsPerLatent[k], 1), dtype=jnp.double)
#         for r in range(n_trials):
#             Z0[k][r,:,0] = jnp.linspace(firstIndPointLoc, trialsLengths[r], nIndPointsPerLatent[k])
#     return Z0


def getEmbeddingSamples(C, d, latents_samples):
    n_trials = len(latents_samples)
    answer = [jnp.matmul(C, latents_samples[r])+d for r in range(n_trials)]
    return answer


def getEmbeddingMeans(C, d, latents_means):
    n_trials = len(latents_means)
    answer = [jnp.matmul(C, latents_means[r])+d for r in range(n_trials)]
    return answer


def getEmbeddingSTDs(C, latents_STDs):
    n_trials = len(latents_STDs)
    answer = [jnp.matmul(C**2, latents_STDs[r]**2).sqrt()
              for r in range(n_trials)]
    return answer


def getCholVecsFromCov(cov):
    cov_chol = []
    for aCov in cov:
        cov_chol.append(svGPFA.utils.miscUtils.chol3D(aCov))
    chol_vecs = getVectorRepOfLowerTrianMatrices(lt_matrices=cov_chol)
    return chol_vecs


def getVectorRepOfLowerTrianMatrices(lt_matrices):
    """Returns vectors containing the lower-triangular elements of the input lower-triangular matrices.

    :parameter lt_matrices: a list of length n_latents, with lt_matrices[k] a tensor of dimension n_trials x n_ind_points x n_ind_points, where lt_matrices[k][r, :, :] is a lower-triangular matrix.
    :type lt_matrices: list

    :return: a list vec_lt_matrices of length n_latents, whith vec_lt_matrices[k] a tensor of dimension (n_trials, n_ind_points*(n_ind_points+1)/2, 0), where vec_lt_matrices[k][r, :, 0] contains the vectorized lower-triangular elements of lt_matrices[k][r, :, :].

    """

    n_latents = len(lt_matrices)
    n_trials = lt_matrices[0].shape[0]

    cholVecs = [[None] for k in range(n_latents)]
    for k in range(n_latents):
        nIndPointsK = lt_matrices[k].shape[1]
        tril_indices = jnp.tril_indices(nIndPointsK)
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        # cholVecs[k] = jnp.empty((n_trials, Pk, 1), dtype=np.float64)
        cholVecs[k] = jnp.empty((n_trials, Pk, 1))
        for r in range(n_trials):
            cholKR = lt_matrices[k][r, :, :]
            cholKRVec = cholKR[tril_indices[0], tril_indices[1]]
            # cholVecs[k][r, :, 0] = cholKRVec
            cholVecs[k] = cholVecs[k].at[r, :, 0].set(cholKRVec)
    return cholVecs

