
import pdb
import torch
from utils import clock

class SVEM:

    # @clock
    def maximize(self, model, measurements, kernels, initialParams, quadParams, 
                 optimParams):
        defaultOptimParams = {"emMaxNIter":20,
                              "eStepMaxNIter":100, 
                              "eStepTol":1e-3, 
                              "eStepLR":1, 
                              "eStepNIterDisplay":10, 
                              "mStepModelParamsMaxNIter":100, 
                              "mStepModelParamsTol":1e-3, 
                              "mStepModelParamsLR":1,  
                              "mStepModelParamsNIterDisplay":10, 
                              "mStepKernelParamsMaxNIter":100, 
                              "mStepKernelParamsTol":1e-3, 
                              "mStepKernelParamsLR":1e-5, 
                              "mStepKernelParamsNIterDisplay":10, 
                              "mStepIndPointsMaxNIter":100, 
                              "mStepIndPointsTol":1e-3, 
                              "mStepIndPointsLR":1, 
                              "mStepIndPointsNIterDisplay":10, 
                              "verbose":True}
        optimParams = {**defaultOptimParams, **optimParams}
        model.setMeasurements(measurements=measurements)
        model.setKernels(kernels=kernels)
        model.setInitialParams(initialParams=initialParams)
        model.setQuadParams(quadParams=quadParams)
        model.buildKernelsMatrices()

        iter = 0
        lowerBoundHist = []
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
            iter += 1
            lowerBoundHist.append(-maxRes['lowerBound'])
        return lowerBoundHist

    def _eStep(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getSVPosteriorOnIndPointsParams()
        evalFunc = model.eval
        optimizer = torch.optim.LBFGS(x)
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, maxNIter=maxNIter, tol=tol, optimizer=optimizer, verbose=verbose, nIterDisplay=nIterDisplay)

    def _mStepModelParams(self, model, maxNIter, tol, lr, verbose, 
                          nIterDisplay):
        x = model.getSVEmbeddingParams()
        svPosteriorOnLatentsStats = model.computeSVPosteriorOnLatentsStats()
        evalFunc = lambda: \
            model.evalELLSumAcrossTrialsAndNeurons(
                svPosteriorOnLatentsStats=svPosteriorOnLatentsStats)
        displayFmt = "Step: %d, negative sum of expected log likelihood: %f"
        optimizer = torch.optim.LBFGS(x)
        answer = self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, maxNIter=maxNIter, tol=tol, optimizer=optimizer, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        return answer

    def _mStepKernelParams(self, model, maxNIter, tol, lr, verbose, 
                           nIterDisplay):
        x = model.getKernelsParams()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.Adam(x, lr=lr)
        updateModelFunc = None
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, maxNIter=maxNIter, tol=tol, optimizer=optimizer, verbose=verbose, nIterDisplay=nIterDisplay)

    def _mStepIndPoints(self, model, maxNIter, tol, lr, verbose, nIterDisplay):
        x = model.getIndPointsLocs()
        def evalFunc():
            model.buildKernelsMatrices()
            answer = model.eval()
            return answer
        optimizer = torch.optim.LBFGS(x)
        updateModelFunc = model.buildKernelsMatrices
        return self._setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, maxNIter=maxNIter, tol=tol, optimizer=optimizer, verbose=verbose, nIterDisplay=nIterDisplay)

    def _setupAndMaximizeStep(self, x, evalFunc, maxNIter, tol, lr, verbose, nIterDisplay, optimizer, displayFmt="Step: %d, negative lower bound: %f", updateModelFunc=None):
        for i in range(len(x)):
            x[i].requires_grad = True 
        maxRes = self._maximizeStep(evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def _maximizeStep(self, evalFunc, updateModelFunc, optimizer, maxNIter,
            tol, verbose, nIterDisplay, displayFmt="Step: %d, negative lower bound: %f"):
        prevEval = -float("inf")
        converged = False
        iterCount = 0
        lowerBoundHist = []
        # pdb.set_trace()
        while not converged and iterCount<maxNIter:
            optimizer.zero_grad()
            curEval = -evalFunc()
            if verbose and iterCount%nIterDisplay==0:
                print(displayFmt%(iterCount, curEval))
            lowerBoundHist.append(-curEval)
            if curEval<prevEval and prevEval-curEval<tol:
                converged = True
            else:
                curEval.backward(retain_graph=True)
                optimizer.step()
                if updateModelFunc is not None:
                    updateModelFunc()
            iterCount += 1
        return {"lowerBound": -curEval, "lowerBoundHist": lowerBoundHist, "converged": converged}

