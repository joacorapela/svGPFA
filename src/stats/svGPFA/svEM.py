
import pdb
import torch
import time
# from .utils import clock

class SVEM:

    # @clock
    def maximize(self, model, measurements, initialParams, quadParams, optimParams):
        defaultOptimParams = {"emMaxNIter":20,
                              "eStepMaxNIter":100,
                              "eStepTol":1e-3,
                              "eStepLR":1e-3,
                              "eStepNIterDisplay":10,
                              "mStepModelParamsMaxNIter":100,
                              "mStepModelParamsTol":1e-3,
                              "mStepModelParamsLR":1e-3,
                              "mStepModelParamsNIterDisplay":10,
                              "mStepKernelParamsMaxNIter":100,
                              "mStepKernelParamsTol":1e-3,
                              "mStepKernelParamsLR":1e-5,
                              "mStepKernelParamsNIterDisplay":10,
                              "mStepIndPointsMaxNIter":100,
                              "mStepIndPointsTol":1e-3,
                              "mStepIndPointsLR":1-5,
                              "mStepIndPointsNIterDisplay":10,
                              "verbose":True}
        optimParams = {**defaultOptimParams, **optimParams}
        model.setMeasurements(measurements=measurements)
        model.setInitialParams(initialParams=initialParams)
        model.setQuadParams(quadParams=quadParams)
        model.buildKernelsMatrices()

        iter = 0
        lowerBoundHist = []
        elapsedTimeHist = []
        startTime = time.time()
        while iter<optimParams["emMaxNIter"]:
            print("Iteration %02d, E-Step start"%(iter))
            maxRes = self._eStep(
                model=model,
                maxNIter=optimParams["eStepMaxNIter"],
                tol=optimParams["eStepTol"],
                lr=optimParams["eStepLR"],
                verbose=optimParams["verbose"],
                nIterDisplay=optimParams["eStepNIterDisplay"])
            print("Iteration %02d, E-Step end: %f"%(iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Model Params start"%(iter))
            # pdb.set_trace()
            maxRes = self._mStepModelParams(
                model=model,
                maxNIter=optimParams["mStepModelParamsMaxNIter"],
                tol=optimParams["mStepModelParamsTol"],
                lr=optimParams["mStepModelParamsLR"],
                verbose=optimParams["verbose"],
                nIterDisplay=optimParams["mStepModelParamsNIterDisplay"])
            print("Iteration %02d, M-Step Model Params end: %f"%
                    (iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Kernel Params start"%(iter))
            # pdb.set_trace()
            maxRes = self._mStepKernelParams(
                model=model,
                maxNIter=optimParams["mStepKernelParamsMaxNIter"],
                tol=optimParams["mStepKernelParamsTol"],
                lr=optimParams["mStepKernelParamsLR"],
                verbose=optimParams["verbose"],
                nIterDisplay=optimParams["mStepKernelParamsNIterDisplay"])
            print("Iteration %02d, M-Step Kernel Params end: %f"%
                    (iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Ind Points start"%(iter))
            # pdb.set_trace()
            maxRes = self._mStepIndPoints(
                model=model,
                maxNIter=optimParams["mStepIndPointsMaxNIter"],
                tol=optimParams["mStepIndPointsTol"],
                lr=optimParams["mStepIndPointsLR"],
                verbose=optimParams["verbose"],
                nIterDisplay=optimParams["mStepIndPointsNIterDisplay"])
            print("Iteration %02d, M-Step Ind Points end: %f"%
                    (iter, -maxRes['lowerBound']))
            # pdb.set_trace()
            elapsedTimeHist.append(time.time()-startTime)
            iter += 1
            lowerBoundHist.append(maxRes['lowerBound'])
        return lowerBoundHist, elapsedTimeHist

    def _eStep(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        updateModelFunc = None
        optimizer = torch.optim.LBFGS(x, lr=lr)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer= self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)
        return answer

    def _mStepModelParams(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getSVEmbeddingParams()
        svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
        evalFunc = lambda: \
            model.evalELLSumAcrossTrialsAndNeurons(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        updateModelFunc = None
        displayFmt = "Step: %d, negative sum of expected log likelihood: %f"
        optimizer = torch.optim.LBFGS(x, lr=lr)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        return answer

    def _mStepKernelParams(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getKernelsParams()
        evalFunc = model.eval
        updateModelFunc = model.buildKernelsMatrices
        optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)
        return answer

    def _mStepIndPoints(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getIndPointsLocs()
        evalFunc = model.eval
        updateModelFunc = model.buildKernelsMatrices
        optimizer = torch.optim.LBFGS(x, lr=lr)
        # optimizer = torch.optim.Adam(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)
        return answer

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer, maxNIter, tol, verbose, nIterDisplay, displayFmt="Step: %d, negative lower bound: %f", updateModelFunc=None):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, updateModelFunc, optimizer, maxNIter, tol, verbose, nIterDisplay, displayFmt="Step: %d, negative lower bound: %f"):
        iterCount = 0
        lowerBoundHist = []
        curEval = None
        converged = False
        while not converged and iterCount<maxNIter:
            def closure():
                # details on this closure at http://sagecal.sourceforge.net/pytorch/index.html
                nonlocal curEval

                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                curEval = -evalFunc()
                if verbose and iterCount%nIterDisplay==0:
                    print(displayFmt%(iterCount, curEval))
                if curEval.requires_grad:
                    curEval.backward(retain_graph=True)
                # print("inside closure curEval={:f}".format(curEval))
                return curEval

            prevEval = curEval
            optimizer.step(closure)
            if updateModelFunc is not None:
                updateModelFunc()
            # print("outside closure curEval={:f}".format(curEval))
            if iterCount>1 and curEval<prevEval and prevEval-curEval<tol:
                converged = True
            lowerBoundHist.append(-curEval.item())
            iterCount += 1

        return {"lowerBound": -curEval.item(), "lowerBoundHist": lowerBoundHist, "converged": converged}
