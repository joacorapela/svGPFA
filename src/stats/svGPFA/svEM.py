
import pdb
import sys
import io
import time
import math
import pickle
import numpy as np
import torch

class TerminationInfo:
    def __init__(self, message):
        self._message = message

    @property
    def message(self):
        return self._message

class ErrorTerminationInfo(TerminationInfo):
    def __init__(self, message, error):
        super.__init__(message=message)
        self._error = error

    @property
    def error(self):
        return self._error

class SVEM:

    def maximize(self, model, optimParams, method="EM",
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
        if method=="EM":
            steps = ["estep", "mstep_embedding", "mstep_kernels", "mstep_indpointslocs"]
            functions_for_steps = {"estep": self._eStep, "mstep_embedding": self._mStepEmbedding, "mstep_kernels": self._mStepKernels, "mstep_indpointslocs": self._mStepIndPointsLocs}
        elif method=="mECM":
            steps = ["estep", "mstep_embedding", "estep", "mstep_kernels", "estep", "mstep_indpointslocs"]
            functions_for_steps = {"estep": self._eStep, "mstep_embedding": self._mStepEmbedding, "estep": self._eStep, "mstep_kernels": self._mStepKernels, "estep": self._eStep, "mstep_indpointslocs": self._mStepIndPointsLocs}
        else:
            raise ValueError("Invalid method={:s}. Supported values are EM and mECM".format(method))
        maxRes = {"lowerBound": -math.inf}
        while iter<optimParams["em_max_iter"]:
            for step in steps:
                if optimParams["{:s}_estimate".format(step)]:
                    message = "Iteration {:02d}, {:s} start: {:f}\n".format(iter, step, maxRes["lowerBound"])
                    if verbose:
                        out.write(message)
                    self._writeToLockedLog(
                        message=message,
                        logLock=logLock,
                        logStream=logStream,
                        logStreamFN=logStreamFN
                    )
                    try:
                        maxRes = functions_for_steps[step](model=model, optimParams=optimParams["{:s}_optim_params".format(step)])
                        message = "Iteration {:02d}, {:s} end: {:f}, niter: {:d}, nfeval: {:d}\n".format(iter, step, maxRes["lowerBound"], maxRes["niter"], maxRes["nfeval"])
                    except:
                        terminationInfo = ErrorTerminationInfo("Error", sys.exc_info)
                        return lowerBoundHist, elapsedTimeHist, terminationInfo
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
        return lowerBoundHist, elapsedTimeHist, terminationInfo

    def _eStep(self, model, optimParams):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, **optimParams)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer)
        return answer

    def _mStepEmbedding(self, model, optimParams):
        x = model.getSVEmbeddingParams()
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

