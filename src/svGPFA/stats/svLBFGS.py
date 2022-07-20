
import pdb
import sys
import io
import time
import math
import pickle
import numpy as np
import torch
import traceback

class TerminationInfo:
    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message

class ErrorTerminationInfo(TerminationInfo):
    def __init__(self, message, error):
        super().__init__(message=message)
        self._error = error

    @property
    def error(self):
        return self._error

class SVLBFGS:

    def maximize(self, model, optimParams, method="EM", getIterationModelParamsFn=None,
                 logLock=None, logStreamFN=None,
                 lowerBoundLock=None, lowerBoundStreamFN=None,
                 latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                 verbose=True, out=sys.stdout,
                 savePartial=False,
                 savePartialFilenamePattern="00000000_{:s}_estimatedModel.pickle",
                ):

        if latentsStreamFN is not None and latentsTimes is None:
            raise RuntimeError("Please specify latentsTime if you want to save latents")

        iter = 0
        if savePartial:
            savePartialFilename = savePartialFilenamePattern.format("initial")
            resultsToSave = {"model": model}
            with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
        lowerBound0 = model.eval()
        lowerBoundHist = [lowerBound0.item()]
        elapsedTimeHist = [0.0]
        startTime = time.time()

        if lowerBoundLock is not None and lowerBoundStreamFN is not None and not lowerBoundLock.is_locked():
            lowerBoundLock.lock()
            with open(lowerBoundStreamFN, 'wb') as f:
                np.save(f, np.array(lowerBoundHist))
            lowerBoundLock.unlock()

        if latentsLock is not None and latentsStreamFN is not None and not latentsLock.is_locked():
            latentsLock.lock()
            muK, varK = model.predictLatents(newTimes=latentsTimes)

            with open(latentsStreamFN, 'wb') as f:
                np.savez(f, iteration=iter, times=latentsTimes.detach().numpy(), muK=muK.detach().numpy(), varK=varK.detach().numpy())
            lowerBoundLock.unlock()
        iter += 1
        logStream = io.StringIO()
        if getIterationModelParamsFn is not None:
            initialModelsParams = getIterationModelParamsFn(model=model)
            iterationsModelParams = torch.empty((optimParams["em_max_iter"]+1, len(initialModelsParams)), dtype=torch.double)
            iterationsModelParams[0,:] = initialModelsParams
        else:
            iterationsModelParams = None
        maxRes = {"lowerBound": lowerBound0}
        while iter<optimParams["em_max_iter"]:
            message = "Iteration {:02d}, start: {:f}\n".format(iter, maxRes["lowerBound"])
            if verbose:
                out.write(message)
            self._writeToLockedLog(
                message=message,
                logLock=logLock,
                logStream=logStream,
                logStreamFN=logStreamFN
            )
            maxRes = self._optimizeAllParams(model=model,
                                             optimParams=optimParams["LBFGS_optim_params"])
            message = "Iteration {:02d}, end: {:f}, niter: {:d}, nfeval: {:d}\n".format(iter, maxRes["lowerBound"], maxRes["niter"], maxRes["nfeval"])
            if verbose:
                out.write(message)
            self._writeToLockedLog(
                message=message,
                logLock=logLock,
                logStream=logStream,
                logStreamFN=logStreamFN
            )
            if savePartial:
                savePartialFilename = savePartialFilenamePattern.format("{:s}{:03d}".format(step, iter))
                resultsToSave = {"model": model}
                with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            if getIterationModelParamsFn is not None:
                iterationsModelParams[iter+1,:] = getIterationModelParamsFn(model=model)
            elapsedTimeHist.append(time.time()-startTime)
            lowerBoundHist.append(maxRes["lowerBound"].item())

            if lowerBoundLock is not None and lowerBoundStreamFN is not None and not lowerBoundLock.is_locked():
                lowerBoundLock.lock()
                with open(lowerBoundStreamFN, 'wb') as f:
                    np.save(f, np.array(lowerBoundHist))
                lowerBoundLock.unlock()

            if latentsLock is not None and latentsStreamFN is not None and not latentsLock.is_locked():
                latentsLock.lock()
                muK, varK = model.predictLatents(newTimes=latentsTimes)

                with open(latentsStreamFN, 'wb') as f:
                    np.savez(f, iteration=iter, times=latentsTimes.detach().numpy(), muK=muK.detach().numpy(), varK=varK.detach().numpy())
                lowerBoundLock.unlock()

            iter += 1
        terminationInfo = TerminationInfo("Maximum number of iterations ({:d}) reached".format(optimParams["em_max_iter"]))
        return lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams

    def _eStep(self, model, optimParams):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _mStepEmbedding(self, model, optimParams):
        x = model.getSVEmbeddingParams()
        # pdb.set_trace()
        # x = [i.contiguous() for i in x]
        svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
        evalFunc = lambda: model.evalELLSumAcrossTrialsAndNeurons(svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        # evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _mStepKernels(self, model, optimParams):
        x = model.getKernelsParams()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        print("Kernel params:", x)
        return answer

    def _mStepIndPointsLocs(self, model, optimParams):
        x = model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _optimizeAllParams(self, model, optimParams):
        x = model.getSVPosteriorOnIndPointsParams() + \
            model.getSVEmbeddingParams() + \
            model.getKernelsParams() + \
            model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            # print("Lower bound: {:.02f}".format(answer))
            return answer
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, optimizer=optimizer)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, optimizer):
        def closure():
            optimizer.zero_grad()
            curEval = -evalFunc()
            curEval.backward(retain_graph=True)
            return curEval
        optimizer.step(closure)
        lowerBound = evalFunc()
        stateOneEpoch = optimizer.state[optimizer._params[0]]
        nfeval = stateOneEpoch["func_evals"]
        niter = stateOneEpoch["n_iter"]
        return {"lowerBound": lowerBound, "nfeval": nfeval, "niter": niter}

    def _writeToLockedLog(self, message, logLock, logStream, logStreamFN):
        logStream.write(message)
        if logLock is not None and not logLock.is_locked():
            logLock.lock()
            with open(logStreamFN, 'a') as f:
                f.write(logStream.getvalue())
            logLock.unlock()
            logStream.truncate(0)
            logStream.seek(0)

