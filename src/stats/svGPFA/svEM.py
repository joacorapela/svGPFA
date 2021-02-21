
import pdb
import sys
import io
import torch
import time
# from .utils import clock
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plot.svGPFA.plotUtils

class SVEM:

    # @clock
    def maximize(self, model, optimParams,
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
        while iter<optimParams["em_max_iter"]:
            if optimParams["estep_estimate"]:
                message = "Iteration %02d, E-Step start\n"%(iter)
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                # begin debug
                # with torch.no_grad():
                #     logLike = model.eval()
                # print(logLike)
                # pdb.set_trace()
                # end debug
                if optimParams["estep_line_search_fn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["estep_line_search_fn"]
                # pdb.set_trace()
                maxRes = self._eStep(
                    model=model,
                    maxIter=optimParams["estep_max_iter"],
                    tol=optimParams["estep_tol"],
                    lr=optimParams["estep_lr"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["estep_niter_display"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN,
                )
                message = "Iteration %02d, E-Step end: %f\n"%(iter, -maxRes['lowerBound'])
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                if savePartial:
                    savePartialFilename = savePartialFilenamePattern.format("eStep{:03d}".format(iter))
                    resultsToSave = {"model": model}
                    with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            # begin debug
            # pdb.set_trace()
            # end debug
            if optimParams["mstep_embedding_estimate"]:
                message = "Iteration %02d, M-Step Model Params start\n"%(iter)
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                # pdb.set_trace()
                if optimParams["mstep_embedding_line_search_fn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["mstep_embedding_line_search_fn"]
                maxRes = self._mStepEmbedding(
                    model=model,
                    maxIter=optimParams["mstep_embedding_max_iter"],
                    tol=optimParams["mstep_embedding_tol"],
                    lr=optimParams["mstep_embedding_lr"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mstep_embedding_niter_display"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN,
                )
                message = "Iteration %02d, M-Step Model Params end: %f\n"%(iter, -maxRes['lowerBound'])
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                if savePartial:
                    savePartialFilename = savePartialFilenamePattern.format("mStepEmbedding{:03d}".format(iter))
                    resultsToSave = {"model": model}
                    with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            # begin debug
            # pdb.set_trace()
            # end debug
            if optimParams["mstep_kernels_estimate"]:
                message = "Iteration %02d, M-Step Kernel Params start\n"%(iter)
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                # pdb.set_trace()
                if optimParams["mstep_kernels_line_search_fn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["mstep_kernels_line_search_fn"]
                maxRes = self._mStepKernels(
                    model=model,
                    maxIter=optimParams["mstep_kernels_max_iter"],
                    tol=optimParams["mstep_kernels_tol"],
                    lr=optimParams["mstep_kernels_lr"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mstep_kernels_niter_display"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN,
                )
                message = "Iteration %02d, M-Step Kernel Params end: %f\n"%(iter, -maxRes['lowerBound'])
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                if savePartial:
                    savePartialFilename = savePartialFilenamePattern.format("mStepKernels{:03d}".format(iter))
                    resultsToSave = {"model": model}
                    with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            # begin debug
            # pdb.set_trace()
            # end debug
            if optimParams["mstep_indpointslocs_estimate"]:
                message = "Iteration %02d, M-Step Ind Points start\n"%(iter)
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                # pdb.set_trace()
                if optimParams["mstep_indpointslocs_line_search_fn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["mstep_indpointslocs_line_search_fn"]
                maxRes = self._mStepIndPointsLocs(
                    model=model,
                    maxIter=optimParams["mstep_indpointslocs_max_iter"],
                    tol=optimParams["mstep_indpointslocs_tol"],
                    lr=optimParams["mstep_indpointslocs_lr"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mstep_indpointslocs_niter_display"],
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN,
                )
                message = "Iteration %02d, M-Step Ind Points end: %f\n"%(iter, -maxRes['lowerBound'])
                if verbose:
                    out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
                if savePartial:
                    savePartialFilename = savePartialFilenamePattern.format("mStepIndPoints{:03d}".format(iter))
                    resultsToSave = {"model": model}
                    with open(savePartialFilename, "wb") as f: pickle.dump(resultsToSave, f)
            # begin debug
            # pdb.set_trace()
            # end debug
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
                    np.savez(f, iteration=iter, times=latentsTimes.detach().numpy(), muK=muK.detach().numpy(), varK=varK.detach().numpy())
                lowerBoundLock.unlock()

            iter += 1
            # pdb.set_trace()
        return lowerBoundHist, elapsedTimeHist

    def _eStep(self, model, maxIter, tol, lr, lineSearchFn, verbose, out,
               nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                            optimizer=optimizer,
                                            maxIter=maxIter, tol=tol,
                                            verbose=verbose,
                                            out=out,
                                            nIterDisplay=nIterDisplay,
                                            logLock=logLock,
                                            logStream=logStream,
                                            logStreamFN=logStreamFN,
                                           )
        return answer

    def _mStepEmbedding(self, model, maxIter, tol, lr, lineSearchFn, verbose, out,
                        nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getSVEmbeddingParams()
        svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
        evalFunc = lambda: \
            model.evalELLSumAcrossTrialsAndNeurons(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                            optimizer=optimizer,
                                            maxIter=maxIter, tol=tol,
                                            verbose=verbose,
                                            out=out,
                                            nIterDisplay=nIterDisplay,
                                            logLock=logLock,
                                            logStream=logStream,
                                            logStreamFN=logStreamFN,
                                           )
        # pdb.set_trace()
        return answer

    def _mStepKernels(self, model, maxIter, tol, lr, lineSearchFn, verbose, out,
                      nIterDisplay, logLock, logStream, logStreamFN,
                      minScale=0.75,
                      displayFmt="Step: %02d, negative lower bound: %f\n",
                     ):
        x = model.getKernelsParams()
        # x = [x[0][0]]
        out.write("kernel params {}\n".format(x))
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                            optimizer=optimizer,
                                            maxIter=maxIter, tol=tol,
                                            verbose=verbose,
                                            out=out,
                                            nIterDisplay=nIterDisplay,
                                            logLock=logLock,
                                            logStream=logStream,
                                            logStreamFN=logStreamFN,
                                           )
        return answer

    def _mStepIndPointsLocs(self, model, maxIter, tol, lr, lineSearchFn, verbose, out,
                        nIterDisplay, logLock, logStream, logStreamFN):
        x = model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
                                            optimizer=optimizer,
                                            maxIter=maxIter, tol=tol,
                                            verbose=verbose,
                                            out=out,
                                            nIterDisplay=nIterDisplay,
                                            logLock=logLock,
                                            logStream=logStream,
                                            logStreamFN=logStreamFN, 
                                           )
        # pdb.set_trace()
        return answer

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer, maxIter, tol,
                              verbose, out, nIterDisplay, logLock, logStream,
                              logStreamFN, 
                              displayFmt="Step: %02d, negative lower bound: %f\n", 
                             ):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, optimizer=optimizer,
                                    maxIter=maxIter, tol=tol, verbose=verbose,
                                    out=out,
                                    nIterDisplay=nIterDisplay,
                                    logLock=logLock,
                                    logStream=logStream,
                                    logStreamFN=logStreamFN,
                                    displayFmt=displayFmt,
                                   )
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, optimizer, maxIter, tol, verbose, out,
                      nIterDisplay, logLock, logStream, logStreamFN,
                      displayFmt="Step: %d, negative lower bound: %f\n",
                     ):
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
            if curEval<=prevEval and prevEval-curEval<tol:
                converged = True
            message = displayFmt%(iterCount, curEval)
            if verbose and iterCount%nIterDisplay==0:
                out.write(message)
                self._writeToLockedLog(
                    message=message,
                    logLock=logLock,
                    logStream=logStream,
                    logStreamFN=logStreamFN
                )
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
            logStream.truncate(0)
            logStream.seek(0)
