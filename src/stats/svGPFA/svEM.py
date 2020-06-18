
import pdb
import io
import torch
import time
# from .utils import clock
import numpy as np
import matplotlib.pyplot as plt
import plot.svGPFA.plotUtils

class SVEM:

    # @clock
    def maximize(self, model, measurements, initialParams, quadParams,
                 optimParams, indPointsLocsKMSEpsilon,
                 logLock=None, logStreamFN=None,
                 lowerBoundLock=None, lowerBoundStreamFN=None,
                 latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                ):

        if latentsStreamFN is not None and latentsTimes is None:
            raise RuntimeError("Please specify latentsTime if you want to save latents")

        model.setMeasurements(measurements=measurements)
        model.setInitialParams(initialParams=initialParams)
        model.setQuadParams(quadParams=quadParams)
        model.setIndPointsLocsKMSEpsilon(indPointsLocsKMSEpsilon=indPointsLocsKMSEpsilon)
        model.buildKernelsMatrices()

        iter = 0
        lowerBound0 = model.eval()
        lowerBoundHist = [lowerBound0]
        elapsedTimeHist = []
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
                np.savez(f, iteration=iter+1, times=latentsTimes.numpy(), muK=muK.numpy(), varK=varK.numpy())
            lowerBoundLock.unlock()

        logStream = io.StringIO()
        while iter<optimParams["emMaxIter"]:
            if optimParams["eStepEstimate"]:
                if self._writeToLockedLog(
                    message="Iteration %02d, E-Step start\n"%(iter+1),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
                maxRes = self._eStep(
                    model=model,
                    maxIter=optimParams["eStepMaxIter"],
                    tol=optimParams["eStepTol"],
                    lr=optimParams["eStepLR"],
                    lineSearchFn=optimParams["eStepLineSearchFn"],
                    verbose=optimParams["verbose"],
                    nIterDisplay=optimParams["eStepNIterDisplay"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN)
                if self._writeToLockedLog(
                    message="Iteration %02d, E-Step end: %f\n"%(iter+1, -maxRes['lowerBound']),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
            if optimParams["mStepEmbeddingEstimate"]:
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Model Params start\n"%(iter+1),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
                # pdb.set_trace()
                maxRes = self._mStepEmbedding(
                    model=model,
                    maxIter=optimParams["mStepEmbeddingMaxIter"],
                    tol=optimParams["mStepEmbeddingTol"],
                    lr=optimParams["mStepEmbeddingLR"],
                    lineSearchFn=optimParams["mStepEmbeddingLineSearchFn"],
                    verbose=optimParams["verbose"],
                    nIterDisplay=optimParams["mStepEmbeddingNIterDisplay"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN)
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Model Params end: %f\n"%(iter+1, -maxRes['lowerBound']),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
            if optimParams["mStepKernelsEstimate"]:
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Kernel Params start\n"%(iter+1),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
                # pdb.set_trace()
                maxRes = self._mStepKernels(
                    model=model,
                    maxIter=optimParams["mStepKernelsMaxIter"],
                    tol=optimParams["mStepKernelsTol"],
                    lr=optimParams["mStepKernelsLR"],
                    lineSearchFn=optimParams["mStepKernelsLineSearchFn"],
                    verbose=optimParams["verbose"],
                    nIterDisplay=optimParams["mStepKernelsNIterDisplay"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN)
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Kernel Params end: %f\n"%(iter+1, -maxRes['lowerBound']),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
            if optimParams["mStepIndPointsEstimate"]:
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Ind Points start\n"%(iter+1),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
                # pdb.set_trace()
                maxRes = self._mStepIndPoints(
                    model=model,
                    maxIter=optimParams["mStepIndPointsMaxIter"],
                    tol=optimParams["mStepIndPointsTol"],
                    lr=optimParams["mStepIndPointsLR"],
                    lineSearchFn=optimParams["mStepIndPointsLineSearchFn"],
                    verbose=optimParams["verbose"],
                    nIterDisplay=optimParams["mStepIndPointsNIterDisplay"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN)
                if self._writeToLockedLog(
                    message="Iteration %02d, M-Step Ind Points end: %f\n"%(iter+1, -maxRes['lowerBound']),
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                ):
                    logStream = io.StringIO()
            elapsedTimeHist.append(time.time()-startTime)
            lowerBoundHist.append(maxRes['lowerBound'])

            if lowerBoundLock is not None and lowerBoundStreamFN is not None and not lowerBoundLock.is_locked():
                lowerBoundLock.lock()
                with open(lowerBoundStreamFN, 'wb') as f:
                    np.save(f, np.array(lowerBoundHist))
                lowerBoundLock.unlock()

            if latentsLock is not None and latentsStreamFN is not None and not latentsLock.is_locked():
                latentsLock.lock()
                muK, varK = model.predictLatents(newTimes=latentsTimes)

                with open(latentsStreamFN, 'wb') as f:
                    np.savez(f, iteration=iter+1, times=latentsTimes.numpy(), muK=muK.numpy(), varK=varK.numpy())
                lowerBoundLock.unlock()

            iter += 1
            # pdb.set_trace()
        return lowerBoundHist, elapsedTimeHist

    def _eStep(self, model, maxIter, tol, lr, lineSearchFn, verbose,
               nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                           optimizer=optimizer,
                                           maxIter=maxIter, tol=tol,
                                           verbose=verbose,
                                           nIterDisplay=nIterDisplay,
                                           logLock=logLock,
                                           logStream=logStream,
                                           logStreamFN=logStreamFN)
        return answer

    def _mStepEmbedding(self, model, maxIter, tol, lr, lineSearchFn, verbose, nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                           optimizer=optimizer,
                                           maxIter=maxIter, tol=tol,
                                           verbose=verbose,
                                           nIterDisplay=nIterDisplay,
                                           logLock=logLock,
                                           logStream=logStream,
                                           logStreamFN=logStreamFN)
        return answer

    def _mStepKernels(self, model, maxIter, tol, lr, lineSearchFn, verbose, nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                           optimizer=optimizer,
                                           maxIter=maxIter, tol=tol,
                                           verbose=verbose,
                                           nIterDisplay=nIterDisplay,
                                           logLock=logLock,
                                           logStream=logStream,
                                           logStreamFN=logStreamFN)
        return answer

    def _mStepIndPoints(self, model, maxIter, tol, lr, lineSearchFn, verbose, nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                           optimizer=optimizer,
                                           maxIter=maxIter, tol=tol,
                                           verbose=verbose,
                                           nIterDisplay=nIterDisplay,
                                           logLock=logLock,
                                           logStream=logStream,
                                           logStreamFN=logStreamFN)
        return answer

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer, maxIter, tol,
                              verbose, nIterDisplay, logLock, logStream,
                              logStreamFN, 
                              displayFmt="Step: %d, negative lower bound: %f\n"):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, optimizer=optimizer,
                                    maxIter=maxIter, tol=tol, verbose=verbose,
                                    nIterDisplay=nIterDisplay,
                                    logLock=logLock,
                                    logStream=logStream, logStreamFN=logStreamFN, displayFmt=displayFmt)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, optimizer, maxIter, tol, verbose,
                      nIterDisplay, logLock, logStream, logStreamFN,
                      displayFmt="Step: %d, negative lower bound: %f\n"):
        iterCount = 0
        lowerBoundHist = []
        curEval = torch.tensor([float("inf")])
        converged = False
        while not converged and iterCount<maxIter:
            def closure():
                # details on this closure at http://sagecal.sourceforge.net/pytorch/index.html
                nonlocal curEval

                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                curEval = -evalFunc()
                if curEval.requires_grad:
                    curEval.backward(retain_graph=True)
                return curEval

            prevEval = curEval
            optimizer.step(closure)
            if curEval<prevEval and prevEval-curEval<tol:
                converged = True
#             if verbose and iterCount%nIterDisplay==0:
#                 self._writeToLockedLog(
#                     message=displayFmt%(iterCount, curEval),
#                     logLock=logLock,
#                     logStream=logStream,
#                     logStreamFN=logStreamFN
#                 )
            lowerBoundHist.append(-curEval.item())
            iterCount += 1

        return {"lowerBound": -curEval.item(), "lowerBoundHist": lowerBoundHist, "converged": converged}

    def _writeToLockedLog(self, message, logLock, logStream, logStreamFN):
        logStream.write(message)
        if logLock is not None and not logLock.is_locked():
            logLock.lock()
            with open(logStreamFN, 'a') as f:
                f.write(logStream.getvalue())
            logLock.unlock()
            return True
        return False
