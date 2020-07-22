
import torch
import myMath.utils
import stats.gaussianProcesses.eval

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

