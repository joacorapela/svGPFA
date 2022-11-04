import scipy.io
import math
import numpy as np
import pandas as pd
import scipy
import sklearn.metrics
import numpy as np
import torch
# import matplotlib.pyplot as plt
import warnings

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
                np.logical_and(trial_start_time_rel <= neuron_spikes_times_rel,
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
    U, S, Vh = np.linalg.svd(C)
    orthoMatrix = Vh.T*S
    n_trials = len(latents_means)
    oLatentsMeans = [[] for r in range(n_trials)]
    for r in range(n_trials):
        oLatentsMeans[r] = np.matmul(latents_means[r], orthoMatrix)
    return oLatentsMeans


def getPropSamplesCovered(sample, mean, std, percent=.95):
    if percent==.95:
        factor = 1.96
    else:
        raise ValueError("percent=0.95 is the only option implemented at the moment")
    covered = torch.logical_and(mean-factor*std<=sample, sample<mean+factor*std)
    coverage = torch.count_nonzero(covered)/float(len(covered))
    return coverage

def getCIFs(C, d, latents):
    n_trials = latents.shape[0]
    nLatents = latents.shape[2]
    nSamples = latents.shape[1]
    nNeurons = C.shape[0]
    embeddings = torch.empty((n_trials, nSamples, nNeurons))
    for r in range(n_trials):
        embeddings[r,:,:] = torch.matmul(latents[r,:,:], torch.transpose(C, 0, 1))+d[:,0]
    CIFs = torch.exp(embeddings)
    return CIFs


def computeSpikeRates(trials_times, spikes_times):
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    spikes_rates = torch.empty((n_trials, n_neurons))
    for r in range(n_trials):
        trial_duration = torch.max(trials_times[r])-torch.min(trials_times[r])
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
    # indPointsLocsKMSEpsilon = np.array(indPointsLocsKMSEpsilon)
    mdict = dict(n_trials=n_trials, nNeurons=nNeurons, nLatents=nLatents,
                 C=C.numpy(), d=torch.reshape(input=d, shape=(-1,1)).numpy(),
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
        mdict.update({"qMu_{:d}".format(k): qMu[k].numpy().astype(np.float64)})
        mdict.update({"qSVec_{:d}".format(k): qSVec[k].numpy().astype(np.float64)})
        mdict.update({"qSDiag_{:d}".format(k): qSDiag[k].numpy().astype(np.float64)})
        mdict.update({"kernelsParams_{:d}".format(k): kernelsParams[k].numpy().astype(np.float64)})
        mdict.update({"indPointsLocs_{:d}".format(k): indPointsLocs[k].numpy().astype(np.float64)})
        mdict.update({"qMu_{:d}".format(k): qMu[k].numpy().astype(np.float64)})
        mdict.update({"qSVec_{:d}".format(k): qSVec[k].numpy().astype(np.float64)})
        mdict.update({"qSDiag_{:d}".format(k): qSDiag[k].numpy().astype(np.float64)})
        mdict.update({"latentsTrialsTimes_{:d}".format(k): latentsTrialsTimes[k].numpy().astype(np.float64)})
    for r in range(n_trials):
        for n in range(nNeurons):
            mdict.update({"spikesTimes_{:d}_{:d}".format(r, n): spikesTimes[r][n].numpy().astype(np.float64)})
    scipy.io.savemat(file_name=saveFilename, mdict=mdict)

def getCholFromVec(vec, nIndPoints):
    chol = torch.zeros((nIndPoints, nIndPoints), dtype=torch.double)
    trilIndices = torch.tril_indices(nIndPoints, nIndPoints)
    chol[trilIndices[0,:],trilIndices[1,:]] = vec
    return chol

def buildCovsFromCholVecs(cholVecs):
    K = len(cholVecs)
    R = cholVecs[0].shape[0]
    covs = [[None] for k in range(K)]
    for k in range(K):
        # cholVecs[k] \in n_trials x Pk x 1
        # Pk = ((nIndPointsK+1)*nIndPointsK)/2
        # nIndPointsK = (-1+sqrt(1+8*Pk))/2
        Pk = cholVecs[k].shape[1]
        nIndPointsK = int((-1+math.sqrt(1+8*Pk))/2)
        covs[k] = torch.empty((R, nIndPointsK, nIndPointsK), dtype=torch.double)
        for r in range(R):
            cholKR = getCholFromVec(vec=cholVecs[k][r,:,0], nIndPoints=nIndPointsK)
            covs[k][r,:,:] = torch.matmul(cholKR, torch.transpose(cholKR, 0, 1))
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
        qSVec[k] = torch.empty(n_trials, nIndPointsK, 1, dtype=torch.double)
        qSDiag[k] = torch.empty(n_trials, nIndPointsK, 1, dtype=torch.double)
        for r in range(n_trials):
            qSRSigmaKR = getCholFromVec(vec=qsCholVecs[k][r, :, 0], nIndPoints=nIndPointsK)
            qSigmaKR = torch.matmul(qSRSigmaKR, torch.transpose(qSRSigmaKR, 0, 1))
            qSDiagKR = torch.diag(qSigmaKR)
            qSigmaKR = qSigmaKR - torch.diag(qSDiagKR)
            eValKR, eVecKR = torch.eig(qSigmaKR, eigenvectors=True)
            maxEvalIKR = torch.argmax(eValKR, dim=0)[0]
            qSVecKR = eVecKR[:, maxEvalIKR]*torch.sqrt(eValKR[maxEvalIKR, 0])
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
    Kchol = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
    nTrial = K.shape[0]
    for i in range(nTrial):
        Kchol[i,:,:] = torch.linalg.cholesky(K[i,:,:])
    return Kchol

def pinv3D(K, rcond=1e-15):
    Kpinv = torch.zeros(K.shape, dtype=K.dtype, device=K.device)
    nTrial = K.shape[0]
    for i in range(nTrial):
        Kpinv[i,:,:] = torch.linalg.pinv(K[i,:,:], rcond=rcond)
    return Kpinv


def getLegQuadPointsAndWeights(n_quad, trials_start_times, trials_end_times,
                               dtype=torch.double):
    n_trials = len(trials_start_times)
    assert(n_trials == len(trials_end_times))
    leg_quad_points = torch.empty((n_trials, n_quad, 1), dtype=dtype)
    leg_quad_weights = torch.empty((n_trials, n_quad, 1), dtype=dtype)
    for r in range(n_trials):
        leg_quad_points[r, :, 0], leg_quad_weights[r, :, 0] = \
                gcnu_common.numerical_methods.utils.leggaussVarLimits(
                    n=n_quad, a=trials_start_times[r], b=trials_end_times[r])
    return leg_quad_points, leg_quad_weights


def getTrialsTimes(start_times, end_times, n_steps):
    assert(len(start_times) == len(end_times))
    n_trials = len(start_times)
    trials_times = torch.empty((n_trials, n_steps, 1), dtype=torch.double)
    for r in range(n_trials):
        trials_times[r, :, 0] = torch.linspace(start_times[r], end_times[r],
                                               n_steps)
    return trials_times


def computeSpikeClassificationROC(spikes_times, cif_times, cif_values,
                                  highres_bin_size=1e-3):
    f = scipy.interpolate.interp1d(cif_times, cif_values)
    cif_times_highres = np.arange(cif_times[0], cif_times[-1],
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
        latents_means[r] = torch.empty((nLatents, len(trialsTimes[r])))
        latents_STDs[r] = torch.empty((nLatents, len(trialsTimes[r])))
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
        latents_STDs[r] = torch.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            latents_STDs[r][k,:] = kernels[k].buildKernelMatrixDiag(X=trialsTimes[r]).sqrt()
    return latents_STDs

# def getLatentsMeanFuncsSamples(latents_meansFuncs, trialsTimes, dtype):
#     n_trials = len(latents_meansFuncs)
#     nLatents = len(latents_meansFuncs[0])
#     latents_meansFuncsSamples = [[] for r in range(n_trials)]
#     for r in range(n_trials):
#         latents_meansFuncsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
#         for k in range(nLatents):
#             latents_meansFuncsSamples[r][k,:] = latents_meansFuncs[r][k](t=trialsTimes[r])

def getLatentsSamplesMeansAndSTDsFromSampledMeans(n_trials, sampledMeans, kernels, trialsTimes, latentsGPRegularizationEpsilon, dtype):
    nLatents = len(kernels)
    latents_samples = [[] for r in range(n_trials)]
    latents_means = [[] for r in range(n_trials)]
    latents_STDs = [[] for r in range(n_trials)]

    for r in range(n_trials):
        latents_samples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latents_means[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latents_STDs[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            print("Procesing trial {:d} and latent {:d}".format(r+1, k+1))
            mean = sampledMeans[r,:,k]
            cov = kernels[k].buildKernelMatrix(trialsTimes[r])
            cov = cov + latentsGPRegularizationEpsilon*torch.eye(cov.shape[0])
            std = torch.diag(cov).sqrt()
            mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            sample = torch.from_numpy(mn.rvs())
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

def getDiagIndicesIn3DArray(N, M, device=torch.device("cpu")):
    frameDiagIndices = torch.arange(end=N, device=device)*(N+1)
    frameStartIndices = torch.arange(end=M, device=device)*N**2
    # torch way of computing an outer sum
    diagIndices = (frameDiagIndices.reshape(-1,1)+frameStartIndices).flatten()
    answer, _ = diagIndices.sort()
    return answer

def build3DdiagFromDiagVector(v, N, M):
    assert(len(v)==N*M)
    diagIndices = getDiagIndicesIn3DArray(N=N, M=M)
    D = torch.zeros(M*N*N, dtype=v.dtype, device=v.device)
    D[diagIndices] = v
    reshapedD = D.reshape(shape = (M, N, N))
    return reshapedD

def buildQSigmaFromQSVecAndQSDiag(qSVec, qSDiag):
    nLatents = len(qSVec)
    n_trials = qSVec[0].shape[0]
    qSigma = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndK = qSDiag[k].shape[1]
        # qq \in n_trials x nInd[k] x 1
        qq = qSVec[k].reshape(shape=(n_trials, nIndK, 1))
        # dd \in n_trials x nInd[k] x 1
        nIndKVarRnkK = qSVec[k].shape[1]
        dd = build3DdiagFromDiagVector(v=(qSDiag[k].flatten())**2, M=n_trials, N=nIndKVarRnkK)
        # qSigma[k] \in n_trials x nInd[k] x nInd[k]
        qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
    return qSigma

def getSRQSigmaVec(qSVec, qSDiag):
    nLatents = len(qSVec)
    n_trials = qSVec[0].shape[0]
    qSigma = buildQSigmaFromQSVecAndQSDiag(qSVec=qSVec, qSDiag=qSDiag)
    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = qSigma[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((n_trials, Pk, 1), dtype=torch.double)
        for r in range(n_trials):
            cholKR = torch.linalg.cholesky(qSigma[k][r,:,:])
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:,0] = cholKRVec
    return srQSigmaVec

# def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPointLoc):
#     nLatents = len(nIndPointsPerLatent)
#     n_trials = len(trialsLengths)
#
#     Z0 = [[] for k in range(nLatents)]
#     for k in range(nLatents):
#         Z0[k] = torch.empty((n_trials, nIndPointsPerLatent[k], 1), dtype=torch.double)
#         for r in range(n_trials):
#             Z0[k][r,:,0] = torch.linspace(firstIndPointLoc, trialsLengths[r], nIndPointsPerLatent[k])
#     return Z0


def getEmbeddingSamples(C, d, latents_samples):
    n_trials = len(latents_samples)
    answer = [torch.matmul(C, latents_samples[r])+d for r in range(n_trials)]
    return answer


def getEmbeddingMeans(C, d, latents_means):
    n_trials = len(latents_means)
    answer = [torch.matmul(C, latents_means[r])+d for r in range(n_trials)]
    return answer


def getEmbeddingSTDs(C, latents_STDs):
    n_trials = len(latents_STDs)
    answer = [torch.matmul(C**2, latents_STDs[r]**2).sqrt()
              for r in range(n_trials)]
    return answer


def getSRQSigmaVecsFromKzz(Kzz):
    Kzz_chol = []
    for aKzz in Kzz:
        Kzz_chol.append(svGPFA.utils.miscUtils.chol3D(aKzz))
    answer = getVectorRepOfLowerTrianMatrices(lt_matrices=Kzz_chol)
    return answer


def getVectorRepOfLowerTrianMatrices(lt_matrices):
    """Returns vectors containing the lower-triangular elements of the input lower-triangular matrices.

    :parameter lt_matrices: a list of length n_latents, with lt_matrices[k] a tensor of dimension n_trials x n_ind_points x n_ind_points, where lt_matrices[k][r, :, :] is a lower-triangular matrix.
    :type lt_matrices: list

    :return: a list vec_lt_matrices of length n_latents, whith vec_lt_matrices[k] a tensor of dimension (n_trials, n_ind_points*(n_ind_points+1)/2, 0), where vec_lt_matrices[k][r, :, 0] contains the vectorized lower-triangular elements of lt_matrices[k][r, :, :].

    """

    n_latents = len(lt_matrices)
    n_trials = lt_matrices[0].shape[0]

    srQSigmaVec = [[None] for k in range(n_latents)]
    for k in range(n_latents):
        nIndPointsK = lt_matrices[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((n_trials, Pk, 1), dtype=torch.double)
        for r in range(n_trials):
            cholKR = lt_matrices[k][r, :, :]
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0, :], trilIndices[1, :]]
            srQSigmaVec[k][r, :, 0] = cholKRVec
    return srQSigmaVec



