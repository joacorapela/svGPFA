import pdb
import scipy.io
import math
import numpy as np
import torch
import scipy.stats
# import matplotlib.pyplot as plt

import svGPFA.stats.kernels
import gcnu_common.numerical_methods.utils
import gcnu_common.stats.gaussian_processes.eval


def buildKernels(kernels_types, kernels_params):
    n_latents = len(kernels_types)
    kernels = [None for k in range(n_latents)]

    for i, kernel_type in enumerate(kernels_types):
        if kernels_types[i] == "exponentialQuadratic":
            kernels[i] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
        elif kernels_types[i] == "periodic":
            kernels[i] = svGPFA.stats.kernels.PeriodicKernel()
        else:
            raise ValueError(f"Invalid kernels type: {kernels_types[i]}")
        kernels[i].setParams(kernels_params[i])
    return kernels


def orthonormalizeLatentsMeans(latentsMeans, C):
    U, S, Vh = np.linalg.svd(C)
    orthoMatrix = Vh.T*S
    nTrials = len(latentsMeans)
    oLatentsMeans = [[] for r in range(nTrials)]
    for r in range(nTrials):
        oLatentsMeans[r] = np.matmul(latentsMeans[r], orthoMatrix)
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
    nTrials = latents.shape[0]
    nLatents = latents.shape[2]
    nSamples = latents.shape[1]
    nNeurons = C.shape[0]
    embeddings = torch.empty((nTrials, nSamples, nNeurons))
    for r in range(nTrials):
        embeddings[r,:,:] = torch.matmul(latents[r,:,:], torch.transpose(C, 0, 1))+d[:,0]
    CIFs = torch.exp(embeddings)
    return CIFs

def computeSpikeRates(trialsTimes, spikesTimes):
    nTrials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    spikesRates = torch.empty((nTrials,nNeurons))
    for r in range(nTrials):
        trialDuration = torch.max(trialsTimes[r])-torch.min(trialsTimes[r])
        for n in range(nNeurons):
            spikesRates[r,n] = len(spikesTimes[r][n])/trialDuration
    return spikesRates

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
    nTrials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    nLatents = len(qMu)
    # indPointsLocsKMSEpsilon = np.array(indPointsLocsKMSEpsilon)
    mdict = dict(nTrials=nTrials, nNeurons=nNeurons, nLatents=nLatents,
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
    for r in range(nTrials):
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
        # cholVecs[k] \in nTrials x Pk x 1
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
    nTrials = qsCholVecs[0].shape[0]
    qSVec = [[] for k in range(nLatents)]
    qSDiag = [[] for k in range(nLatents)]
    for k in range(nLatents):
        Pk = qsCholVecs[k].shape[1]
        nIndPointsK = int((-1.0+math.sqrt(1+8*Pk))/2.0)
        qSVec[k] = torch.empty(nTrials, nIndPointsK, 1, dtype=torch.double)
        qSDiag[k] = torch.empty(nTrials, nIndPointsK, 1, dtype=torch.double)
        for r in range(nTrials):
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
    # begin debug
#     print("Waning: debug code on in miscUtils.py:chol3D")
#     nTrial = K.shape[0]
#     for r in range(nTrial):
#         Kr = torch.matmul(K[r,:,:], torch.transpose(K[r,:,:], 0, 1))
#         eigRes = torch.eig(Kr)
#         cNum = eigRes.eigenvalues[0,0]/eigRes.eigenvalues[-1,0]
#         print("Condition number for trial {:d}: {:f}".format(r, cNum))
    # pdb.set_trace()
    # end debug
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
    nTrials = len(trials_start_times)
    assert(nTrials == len(trials_end_times))
    leg_quad_points = torch.empty((nTrials, n_quad, 1), dtype=dtype)
    leg_quad_weights = torch.empty((nTrials, n_quad, 1), dtype=dtype)
    for r in range(nTrials):
        leg_quad_points[r, :, 0], leg_quad_weights[r, :, 0] = \
                gcnu_common.numerical_methods.utils.leggaussVarLimits(
                    n=n_quad, a=trials_start_times[r], b=trials_end_times[r])
    return leg_quad_points, leg_quad_weights


def getTrialsTimes(trialsLengths, dt):
    nTrials = len(trialsLengths)
    trialsTimes = [[] for r in range(nTrials)]
    for r in range(nTrials):
        trialsTimes[r] = torch.linspace(0, trialsLengths[r], round(trialsLengths[r]/dt))
    return trialsTimes

def getLatentsMeansAndSTDs(meansFuncs, kernels, trialsTimes):
    nTrials = len(trialsTimes)
    nLatents = len(kernels)
    latentsMeans = [[] for r in range(nTrials)]
    latentsSTDs = [[] for r in range(nTrials)]

    for r in range(nTrials):
        latentsMeans[r] = torch.empty((nLatents, len(trialsTimes[r])))
        latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            gp = gcnu_common.stats.gaussian_processes.eval.GaussianProcess(mean=meansFuncs[k], kernel=kernels[k])
            latentsMeans[r][k,:] = gp.mean(t=trialsTimes[r])
            latentsSTDs[r][k,:] = gp.std(t=trialsTimes[r])
    return latentsMeans, latentsSTDs

def getLatentsSTDs(kernels, trialsTimes):
    nTrials = len(trialsTimes)
    nLatents = len(kernels)
    latentsSTDs = [[] for r in range(nTrials)]

    for r in range(nTrials):
        latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])))
        for k in range(nLatents):
            latentsSTDs[r][k,:] = kernels[k].buildKernelMatrixDiag(X=trialsTimes[r]).sqrt()
            pdb.set_trace()
    return latentsSTDs

# def getLatentsMeanFuncsSamples(latentsMeansFuncs, trialsTimes, dtype):
#     nTrials = len(latentsMeansFuncs)
#     nLatents = len(latentsMeansFuncs[0])
#     latentsMeansFuncsSamples = [[] for r in range(nTrials)]
#     for r in range(nTrials):
#         latentsMeansFuncsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
#         for k in range(nLatents):
#             latentsMeansFuncsSamples[r][k,:] = latentsMeansFuncs[r][k](t=trialsTimes[r])

def getLatentsSamplesMeansAndSTDsFromSampledMeans(nTrials, sampledMeans, kernels, trialsTimes, latentsGPRegularizationEpsilon, dtype):
    nLatents = len(kernels)
    latentsSamples = [[] for r in range(nTrials)]
    latentsMeans = [[] for r in range(nTrials)]
    latentsSTDs = [[] for r in range(nTrials)]

    for r in range(nTrials):
        latentsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latentsMeans[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            print("Procesing trial {:d} and latent {:d}".format(r+1, k+1))
            mean = sampledMeans[r,:,k]
            cov = kernels[k].buildKernelMatrix(trialsTimes[r])
            cov = cov + latentsGPRegularizationEpsilon*torch.eye(cov.shape[0])
            std = torch.diag(cov).sqrt()
            mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
            sample = torch.from_numpy(mn.rvs())
            latentsSamples[r][k,:] = sample
            latentsMeans[r][k,:] = mean
            latentsSTDs[r][k,:] = std
            # plt.plot(trialsTimes[r], mean, label="mean")
            # plt.plot(trialsTimes[r], sample, label="sample")
            # plt.xlabel("Time (sec)")
            # plt.ylabel("Value")
            # plt.title("Latent {:d}".format(k))
            # plt.legend()
            # plt.show()
    return latentsSamples, latentsMeans, latentsSTDs

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
    nTrials = qSVec[0].shape[0]
    qSigma = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndK = qSDiag[k].shape[1]
        # qq \in nTrials x nInd[k] x 1
        qq = qSVec[k].reshape(shape=(nTrials, nIndK, 1))
        # dd \in nTrials x nInd[k] x 1
        nIndKVarRnkK = qSVec[k].shape[1]
        dd = build3DdiagFromDiagVector(v=(qSDiag[k].flatten())**2, M=nTrials, N=nIndKVarRnkK)
        # qSigma[k] \in nTrials x nInd[k] x nInd[k]
        qSigma[k] = torch.matmul(qq, torch.transpose(a=qq, dim0=1, dim1=2)) + dd
    return qSigma

def getSRQSigmaVec(qSVec, qSDiag):
    nLatents = len(qSVec)
    nTrials = qSVec[0].shape[0]
    qSigma = buildQSigmaFromQSVecAndQSDiag(qSVec=qSVec, qSDiag=qSDiag)
    srQSigmaVec = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        nIndPointsK = qSigma[k].shape[1]
        Pk = int((nIndPointsK+1)*nIndPointsK/2)
        srQSigmaVec[k] = torch.empty((nTrials, Pk, 1), dtype=torch.double)
        for r in range(nTrials):
            cholKR = torch.linalg.cholesky(qSigma[k][r,:,:])
            trilIndices = torch.tril_indices(nIndPointsK, nIndPointsK)
            cholKRVec = cholKR[trilIndices[0,:], trilIndices[1,:]]
            srQSigmaVec[k][r,:,0] = cholKRVec
    return srQSigmaVec

# def getIndPointLocs0(nIndPointsPerLatent, trialsLengths, firstIndPointLoc):
#     nLatents = len(nIndPointsPerLatent)
#     nTrials = len(trialsLengths)
#
#     Z0 = [[] for k in range(nLatents)]
#     for k in range(nLatents):
#         Z0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
#         for r in range(nTrials):
#             Z0[k][r,:,0] = torch.linspace(firstIndPointLoc, trialsLengths[r], nIndPointsPerLatent[k])
#     return Z0

def getEmbeddingSamples(C, d, latentsSamples):
    nTrials = len(latentsSamples)
    answer = [torch.matmul(C, latentsSamples[r])+d for r in range(nTrials)]
    return answer

def getEmbeddingMeans(C, d, latentsMeans):
    nTrials = len(latentsMeans)
    answer = [torch.matmul(C, latentsMeans[r])+d for r in range(nTrials)]
    return answer

def getEmbeddingSTDs(C, latentsSTDs):
    nTrials = len(latentsSTDs)
    answer = [torch.matmul(C**2, latentsSTDs[r]**2).sqrt() for r in range(nTrials)]
    return answer


def getSRQSigmaVecsFromKzz(Kzz):
    Kzz_chol = []
    for aKzz in Kzz:
        Kzz_chol.append(svGPFA.utils.miscUtils.chol3D(aKzz))
    answer = getVectorRepOfLowerTrianMatrices(lt_matrices=Kzz_chol)
    return answer


def getVectorRepOfLowerTrianMatrices(lt_matrices):
    """Returns vectors containing the lower-triangular elements of the input lower-triangular matrices.

    :parameter lt_matrices: a list of length n_latents, with lt_matrices[k] a tensor of dimension n_trials x nIndPoints x nIndPoints, where lt_matrices[k][r, :, :] is a lower-triangular matrix.
    :type lt_matrices: list

    :return: a list srQSigmaVec of length n_latents, whith srQSigmaVec[k] a tensor of dimension n_trials x (nIndPoints+1)*nIndPoints/2 x 0, where srQSigmaVec[k][r, :, 0] contains the lower-triangular elements of lt_matrices[k][r, :, :]
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



