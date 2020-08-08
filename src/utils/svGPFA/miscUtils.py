import pdb
import scipy.io
import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
import myMath.utils
import stats.gaussianProcesses.eval

def getCIFs(C, d, latents):
    nTrials = latents.shape[0]
    nLatents = latents.shape[2]
    nSamples = latents.shape[1]
    nNeurons = C.shape[0]
    embeddings = torch.empty((nTrials, nSamples, nNeurons))
    for r in range(nTrials):
        embeddings[r,:,:] = torch.matmul(latents[r,:,:], torch.transpose(C, 0, 1))+d[:,0]
    CIFs = torch.exp(embeddings)
    return(CIFs)

def computeSpikeRates(trialsTimes, spikesTimes):
    nTrials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    spikesRates = torch.empty((nTrials,nNeurons))
    for r in range(nTrials):
        trialDuration = torch.max(trialsTimes[r])-torch.min(trialsTimes[r])
        for n in range(nNeurons):
            spikesRates[r,n] = len(spikesTimes[r][n])/trialDuration
    return spikesRates

def saveDataForMatlabEstimations(qMu0, qSVec0, qSDiag0, C0, d0,
                                 indPointsLocs0,
                                 legQuadPoints, legQuadWeights,
                                 kernelsTypes, kernelsParams0,
                                 spikesTimes,
                                 indPointsLocsKMSEpsilon, trialsLengths,
                                 emMaxIter, eStepMaxIter, mStepEmbeddingMaxIter,
                                 mStepKernelsMaxIter, mStepIndPointsMaxIter,
                                 saveFilename):
    nTrials = len(spikesTimes)
    nNeurons = len(spikesTimes[0])
    nLatents = len(qMu0)
    # indPointsLocsKMSEpsilon = np.array(indPointsLocsKMSEpsilon)
    mdict = dict(nTrials=nTrials, nNeurons=nNeurons, nLatents=nLatents,
                 C0=C0.numpy(), d0=torch.reshape(input=d0, shape=(-1,1)).numpy(),
                 legQuadPoints=legQuadPoints.numpy(),
                 legQuadWeights=legQuadWeights.numpy(),
                 indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon,
                 trialsLengths=trialsLengths,
                 emMaxIter=emMaxIter, eStepMaxIter=eStepMaxIter,
                 mStepEmbeddingMaxIter=mStepEmbeddingMaxIter,
                 mStepKernelsMaxIter=mStepKernelsMaxIter,
                 mStepIndPointsMaxIter=mStepIndPointsMaxIter)
    for k in range(nLatents):
        mdict.update({"kernelType_{:d}".format(k): kernelsTypes[k]})
        mdict.update({"qMu0_{:d}".format(k): qMu0[k].numpy().astype(np.float64)})
        mdict.update({"qSVec0_{:d}".format(k): qSVec0[k].numpy().astype(np.float64)})
        mdict.update({"qSDiag0_{:d}".format(k): qSDiag0[k].numpy().astype(np.float64)})
        mdict.update({"kernelsParams0_{:d}".format(k): kernelsParams0[k].numpy().astype(np.float64)})
        mdict.update({"indPointsLocs0_{:d}".format(k): indPointsLocs0[k].numpy().astype(np.float64)})
        mdict.update({"qMu0_{:d}".format(k): qMu0[k].numpy().astype(np.float64)})
        mdict.update({"qSVec0_{:d}".format(k): qSVec0[k].numpy().astype(np.float64)})
        mdict.update({"qSDiag0_{:d}".format(k): qSDiag0[k].numpy().astype(np.float64)})
    for r in range(nTrials):
        for n in range(nNeurons):
            mdict.update({"spikesTimes_{:d}_{:d}".format(r, n): spikesTimes[r][n].numpy().astype(np.float64)})
    scipy.io.savemat(file_name=saveFilename, mdict=mdict)

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
    for i in range(K.shape[0]):
        Kchol[i,:,:] = torch.cholesky(K[i,:,:])
    return Kchol

def getLegQuadPointsAndWeights(nQuad, trialsLengths, dtype=torch.double):
    nTrials = len(trialsLengths)
    legQuadPoints = torch.empty((nTrials, nQuad, 1), dtype=dtype)
    legQuadWeights = torch.empty((nTrials, nQuad, 1), dtype=dtype)
    for r in range(nTrials):
        legQuadPoints[r,:,0], legQuadWeights[r,:,0] = myMath.utils.leggaussVarLimits(n=nQuad, a=0, b=trialsLengths[r])
    return legQuadPoints, legQuadWeights

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
            gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[k], kernel=kernels[k])
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
            plt.plot(trialsTimes[r], mean, label="mean")
            plt.plot(trialsTimes[r], sample, label="sample")
            plt.xlabel("Time (sec)")
            plt.ylabel("Value")
            plt.title("Latent {:d}".format(k))
            plt.legend()
            plt.show()
    return latentsSamples, latentsMeans, latentsSTDs

def getLatentsSamplesMeansAndSTDs(nTrials, meansFuncs, kernels, trialsTimes, latentsGPRegularizationEpsilon, dtype):
    nLatents = len(kernels)
    latentsSamples = [[] for r in range(nTrials)]
    latentsMeans = [[] for r in range(nTrials)]
    latentsSTDs = [[] for r in range(nTrials)]

    for r in range(nTrials):
        latentsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latentsMeans[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            print("Processing trial {:d} and latent {:d}".format(r, k))
            gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[k], kernel=kernels[k])
            sample, mean, cov  = gp.eval(t=trialsTimes[r], regularization=latentsGPRegularizationEpsilon)
            latentsSamples[r][k,:] = sample
            latentsMeans[r][k,:] = mean
            latentsSTDs[r][k,:] = torch.diag(cov).sqrt()
    return latentsSamples, latentsMeans, latentsSTDs

