
import torch
import stats.gaussianProcesses.eval

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

def getLatentsMeanFuncsSamples(latentsMeansFuncs, trialsTimes, dtype):
    nTrials = len(latentsMeansFuncs)
    nLatents = len(latentsMeansFuncs[0])
    latentsMeansFuncsSamples = [[] for r in range(nTrials)]
    for r in range(nTrials):
        latentsMeansFuncsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            latentsMeansFuncsSamples[r][k,:] = latentsMeansFuncs[r][k](t=trialsTimes[r])
    return latentsMeansFuncsSamples

def getLatentsSamples(meansFuncs, kernels, trialsTimes, gpRegularization, dtype):
    nTrials = len(kernels)
    nLatents = len(kernels[0])
    latentsSamples = [[] for r in range(nTrials)]

    for r in range(nTrials):
        print("Procesing trial {:d}".format(r))
        latentsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
        for k in range(nLatents):
            print("Procesing latent {:d}".format(k))
            gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[r][k], kernel=kernels[r][k])
            latentsSamples[r][k,:] = gp.eval(t=trialsTimes[r], regularization=gpRegularization,)
    return latentsSamples

