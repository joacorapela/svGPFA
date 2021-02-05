
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
    def maximize(self, model, measurements, initialParams, quadParams,
                 optimParams, indPointsLocsKMSRegEpsilon,
                 logLock=None, logStreamFN=None,
                 lowerBoundLock=None, lowerBoundStreamFN=None,
                 latentsTimes=None, latentsLock=None, latentsStreamFN=None,
                 verbose=True, out=sys.stdout,
                 savePartial=False,
                 savePartialFilenamePattern="00000000_{:s}_estimatedModel.pickle",
                ):

        if latentsStreamFN is not None and latentsTimes is None:
            raise RuntimeError("Please specify latentsTime if you want to save latents")

        model.setMeasurements(measurements=measurements)
        model.setInitialParams(initialParams=initialParams)
        model.setQuadParams(quadParams=quadParams)
        model.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
        model.buildKernelsMatrices()

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
        while iter<optimParams["emMaxIter"]:
            if optimParams["eStepEstimate"]:
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
                if optimParams["eStepLineSearchFn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["eStepLineSearchFn"]
                # pdb.set_trace()
                maxRes = self._eStep(
                    model=model,
                    maxIter=optimParams["eStepMaxIter"],
                    tol=optimParams["eStepTol"],
                    lr=optimParams["eStepLR"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["eStepNIterDisplay"],
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
            if optimParams["mStepEmbeddingEstimate"]:
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
                if optimParams["eStepLineSearchFn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["eStepLineSearchFn"]
                maxRes = self._mStepEmbedding(
                    model=model,
                    maxIter=optimParams["mStepEmbeddingMaxIter"],
                    tol=optimParams["mStepEmbeddingTol"],
                    lr=optimParams["mStepEmbeddingLR"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mStepEmbeddingNIterDisplay"],
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
            if optimParams["mStepKernelsEstimate"]:
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
                if optimParams["eStepLineSearchFn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["eStepLineSearchFn"]
                maxRes = self._mStepKernels(
                    model=model,
                    maxIter=optimParams["mStepKernelsMaxIter"],
                    tol=optimParams["mStepKernelsTol"],
                    lr=optimParams["mStepKernelsLR"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mStepKernelsNIterDisplay"],
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
            if optimParams["mStepIndPointsEstimate"]:
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
                if optimParams["eStepLineSearchFn"]=="None":
                    lineSearchFn = None
                else:
                    lineSearchFn = optimParams["eStepLineSearchFn"]
                maxRes = self._mStepIndPointsLocs(
                    model=model,
                    maxIter=optimParams["mStepIndPointsMaxIter"],
                    tol=optimParams["mStepIndPointsTol"],
                    lr=optimParams["mStepIndPointsLR"],
                    lineSearchFn=lineSearchFn,
                    verbose=optimParams["verbose"],
                    out=out,
                    nIterDisplay=optimParams["mStepIndPointsNIterDisplay"],
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
        # out.write("*** Bug in _mStepKernels ***\n")
        # x = [x[0][0]]
        out.write("kernel params {}\n".format(x))
        # begin debug periodic kernel
#         with torch.no_grad():
#             import pandas as pd
#             import plotly.io as pio
#             import plotly.express as px
#             import plotly.graph_objs as go
#             startLB = model.eval()
#             xStart = x[0].clone()
#             displacements = np.arange(-4.0, 4.0, .1)
#             uniqueLengthscales = x[0][0].item() + displacements
#             uniquePeriods = x[0][1].item() + displacements
#             allLengthscales = []
#             allPeriods = []
#             allLowerBounds = []
#             for ls in uniqueLengthscales:
#                 for p in uniquePeriods:
#                     allLengthscales.append(ls)
#                     allPeriods.append(p)
#                     x[0][0] = ls
#                     x[0][1] = p
#                     model.buildKernelsMatrices()
#                     lowerBound = model.eval().item()
#                     allLowerBounds.append(lowerBound)
#                     # print("Lower bound for lengthscale {:.02f} and period {:.02f} is {:.02f}".format(ls, p, lowerBound))
#             x[0][0] = xStart[0]
#             x[0][1] = xStart[1]
#             data = {"lenghtscale": allLengthscales, "period": allPeriods, "lowerBound": allLowerBounds}
#             df = pd.DataFrame(data)
#             fig = px.scatter_3d(df, x='lenghtscale', y='period', z='lowerBound')
#         pdb.set_trace()
        # end debug periodic kernel
        # begin debug exponential quadratic kernel
#         with torch.no_grad():
#             import pandas as pd
#             import plotly.io as pio
#             import plotly.express as px
#             import plotly.graph_objs as go
#             startLB = model.eval()
#             xStart = x[0].clone()
#             displacements = np.arange(-4.0, 4.0, .1)
#             uniqueLengthscales = x[0][0].item() + displacements
#             allLengthscales = []
#             allLowerBounds = []
#             for ls in uniqueLengthscales:
#                 allLengthscales.append(ls)
#                 x[0][0] = ls
#                 model.buildKernelsMatrices()
#                 lowerBound = model.eval().item()
#                 allLowerBounds.append(lowerBound)
#                 # print("Lower bound for lengthscale {:.02f} and period {:.02f} is {:.02f}".format(ls, p, lowerBound))
#             x[0][0] = xStart[0]
#             data = {"lenghtscale": allLengthscales, "lowerBound": allLowerBounds}
#             df = pd.DataFrame(data)
#             fig = px.scatter(df, x='lenghtscale', y='lowerBound')
#         pdb.set_trace()
        # end debug exponential quadratic kernel
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, lr=lr, line_search_fn=lineSearchFn)
#         optimizer = torch.optim.Adam(x, lr=lr)
#         answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc,
#                                             optimizer=optimizer,
#                                             maxIter=maxIter, tol=tol,
#                                             verbose=verbose,
#                                             out=out,
#                                             nIterDisplay=nIterDisplay,
#                                             logLock=logLock,
#                                             logStream=logStream,
#                                             logStreamFN=logStreamFN,
#                                            )
        for i in range(len(x)):
            x[i].requires_grad = True
        iterCount = 0
        lowerBoundHist = []
        curEval = torch.tensor([float("inf")])
        converged = False
        def closure():
            # details on this closure at http://sagecal.sourceforge.net/pytorch/index.html
            nonlocal curEval

            if torch.is_grad_enabled():
                optimizer.zero_grad()
            curEval = -evalFunc()
            if curEval.requires_grad:
                curEval.backward(retain_graph=True)
            return curEval

        while not converged and iterCount<maxIter:
            prevEval = curEval
            optimizer.step(closure)
            with torch.no_grad():
                x[0].clamp_(minScale)
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

        answer = {"lowerBound": -curEval.item(), "lowerBoundHist": lowerBoundHist, "converged": converged}

        for i in range(len(x)):
            x[i].requires_grad = False

        # begin debug periodic kernel
#         with torch.no_grad():
#             endLB = model.eval()
#             xStart = xStart.numpy()
#             xEnd = model.getKernelsParams()[0].clone().numpy()
#             fig.add_trace(go.Scatter3d(x=[xStart[0],xEnd[0]], y=[xStart[1],xEnd[1]], z=[startLB, endLB], type="scatter3d", text=["start","end"], mode="text"))
#             fig.update_layout(scene = dict(zaxis = dict(range=[df.lowerBound.max()-1000,df.lowerBound.max()],),),)
#             fig.write_image("/tmp/tmp.png")
#             fig.write_html("/tmp/tmp.html")
#             pio.renderers.default = "browser"
#             fig.show()
#         pdb.set_trace()
        # end debug periodic kernel
        # begin debug exponential quadratic kernel
#         with torch.no_grad():
#             endLB = model.eval()
#             xStart = xStart.numpy()
#             xEnd = model.getKernelsParams()[0].clone().numpy()
#             fig.add_trace(go.Scatter(x=[xStart[0],xEnd[0]], y=[startLB, endLB], type="scatter", text=["start","end"], mode="text"))
#             fig.update_layout(scene = dict(yaxis = dict(range=[df.lowerBound.max()-1000,df.lowerBound.max()],),),)
#             fig.write_image("/tmp/tmp.png")
#             fig.write_html("/tmp/tmp.html")
#             pio.renderers.default = "browser"
#             fig.show()
#         pdb.set_trace()
        # end debug exponential quadratic kernel
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
