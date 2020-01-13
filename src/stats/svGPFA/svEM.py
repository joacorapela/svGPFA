
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
                              "eStepLR":1,
                              "eStepNIterDisplay":1,
                              "mStepModelParamsMaxNIter":100,
                              "mStepModelParamsTol":1e-3,
                              "mStepModelParamsLR":1,
                              "mStepModelParamsNIterDisplay":1,
                              "mStepKernelParamsMaxNIter":100,
                              "mStepKernelParamsTol":1e-3,
                              "mStepKernelParamsLR":1e-5,
                              "mStepKernelParamsNIterDisplay":1,
                              "mStepIndPointsMaxNIter":100,
                              "mStepIndPointsTol":1e-3,
                              "mStepIndPointsLR":1,
                              "mStepIndPointsNIterDisplay":1,
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
            maxRes = self._mStepIndPoints(
                model=model,
                maxNIter=optimParams["mStepIndPointsMaxNIter"],
                tol=optimParams["mStepIndPointsTol"],
                lr=optimParams["mStepIndPointsLR"],
                verbose=optimParams["verbose"],
                nIterDisplay=optimParams["mStepIndPointsNIterDisplay"])
            print("Iteration %02d, M-Step Ind Points end: %f"%
                    (iter, -maxRes['lowerBound']))
            elapsedTimeHist.append(time.time()-startTime)
            iter += 1
            lowerBoundHist.append(maxRes['lowerBound'])
        return lowerBoundHist, elapsedTimeHist

    def _eStep(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x, lr=lr)
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)

    def _mStepModelParams(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getSVEmbeddingParams()
        svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
        evalFunc = lambda: \
            model.evalELLSumAcrossTrialsAndNeurons(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        displayFmt = "Step: %d, negative sum of expected log likelihood: %f"
        optimizer = torch.optim.LBFGS(x, lr=lr)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        return answer

    def _mStepKernelParams(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getKernelsParams()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.Adam(x, lr=lr)
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)

    def _mStepIndPoints(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x, lr=lr)
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay)

    def _setupAndMaximizeStep(self, x, evalFunc, optimizer, maxNIter, tol, verbose, nIterDisplay, displayFmt="Step: %d, negative lower bound: %f"):
        for i in range(len(x)):
            x[i].requires_grad = True
        maxRes = self._maximizeStep(evalFunc=evalFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, optimizer, maxNIter, tol, verbose, nIterDisplay, displayFmt="Step: %d, negative lower bound: %f"):
        iterCount = 0
        lowerBoundHist = []
        curEval = torch.tensor([float("inf")])
        converged = False
        while not converged and iterCount<maxNIter:
            def closure():
                # details on this closure at http://sagecal.sourceforge.net/pytorch/index.html
                nonlocal curEval
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                # pdb.set_trace()
                curEval = -evalFunc()
                if curEval.requires_grad:
                    curEval.backward(retain_graph=True)
                # print("inside closure curEval={:f}".format(curEval))
                return curEval

            prevEval = curEval
            optimizer.step(closure)
            # print("outside closure curEval={:f}".format(curEval))
            # pdb.set_trace()
            if curEval<prevEval and prevEval-curEval<tol:
                converged = True
            if verbose and iterCount%nIterDisplay==0:
                print(displayFmt%(iterCount, curEval))
            lowerBoundHist.append(-curEval)
            iterCount += 1

        return {"lowerBound": -curEval, "lowerBoundHist": lowerBoundHist, "converged": converged}
