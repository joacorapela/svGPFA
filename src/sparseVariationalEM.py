
import pdb
import torch

class SparseVariationalEM:

    def __init__(self, lowerBound, eLL, kernelMatricesStore):
        self.__lowerBound = lowerBound
        self.__eLL = eLL
        self.__kernelMatricesStore= kernelMatricesStore

    def maximize(self, emMaxNIter=20, 
            eStepMaxNIter=100, eStepTol=1e-3, eStepLR=1e-3, eStepNIterDisplay=10,
            mStepModelParamsMaxNIter=100, mStepModelParamsTol=1e-3, mStepModelParamsLR=1e-3,  mStepModelParamsNIterDisplay=10,
            mStepKernelParamsMaxNIter=100, mStepKernelParamsTol=1e-3, mStepKernelParamsLR=1e-5, mStepKernelParamsNIterDisplay=10,
            mStepInducingPointsMaxNIter=100, mStepInducingPointsTol=1e-3, mStepInducingPointsLR=1e-3, mStepInducingPointsNIterDisplay=10,
            verbose=True):
        iter = 0
        while iter<emMaxNIter:
            print("Iteration %02d, E-Step start"%(iter))
            maxRes = self.__eStep(maxNIter=eStepMaxNIter, tol=eStepTol, lr=eStepLR, verbose=verbose, nIterDisplay=eStepNIterDisplay)
            print("Iteration %02d, E-Step end: %f"%(iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Model Params start"%(iter))
            maxRes = self.__mStepModelParams(maxNIter=mStepModelParamsMaxNIter, tol=mStepModelParamsTol, lr=mStepModelParamsLR, verbose=verbose, nIterDisplay=mStepModelParamsNIterDisplay)
            print("Iteration %02d, M-Step Model Params end: %f"%(iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Kernel Params start"%(iter))
            maxRes = self.__mStepKernelParams(maxNIter=mStepKernelParamsMaxNIter, tol=mStepKernelParamsTol, lr=mStepKernelParamsLR, verbose=verbose, nIterDisplay=mStepKernelParamsNIterDisplay)
            print("Iteration %02d, M-Step Kernel Params end: %f"%(iter, -maxRes['lowerBound']))
            print("Iteration %02d, M-Step Inducing Points start"%(iter))
            maxRes = self.__mStepInducingPoints(maxNIter=mStepInducingPointsMaxNIter, tol=mStepInducingPointsTol, lr=mStepInducingPointsLR, verbose=verbose, nIterDisplay=mStepInducingPointsNIterDisplay)
            print("Iteration %02d, M-Step Inducing Points end: %f"%(iter, -maxRes['lowerBound']))
            iter += 1
        return maxRes

    def __eStep(self, maxNIter, tol, lr, verbose, nIterDisplay):
        x = self.__lowerBound.getApproxPosteriorForHParams()
        evalFunc = self.__lowerBound.eval
        return self.__setupAndMaximizeStep(x=x, evalFunc=evalFunc, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose, nIterDisplay=nIterDisplay)

    def __mStepModelParams(self, maxNIter, tol, lr, verbose, nIterDisplay):
        # print("Negative lower bound at start of __mStepModelParams: %f"%(-self.__lowerBound.eval()))
        x = self.__eLL.getModelParams()
        kFactors = self.__eLL.buildKFactors()
        evalFunc = lambda: self.__eLL.evalSumAcrossTrialsAndNeurons(kFactors=kFactors)
        # evalFunc = self.__lowerBound.eval
        displayFmt = "Iteration: %d, negative sum of expected log likelihood: %f"
        answer = self.__setupAndMaximizeStep(x=x, evalFunc=evalFunc, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        # print("Negative lower bound at end of __mStepModelParams: %f"%(-self.__lowerBound.eval()))
        return answer

    def __mStepKernelParams(self, maxNIter, tol, lr, verbose, nIterDisplay):
        x = self.__kernelMatricesStore.getKernelsVariableParameters()
        def evalFunc():
            self.__kernelMatricesStore.buildKernelMatrices()
            answer = self.__lowerBound.eval()
            return answer
        # updateModelFunc = self.__kernelMatricesStore.buildKernelMatrices
        updateModelFunc = None
        return self.__setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose, nIterDisplay=nIterDisplay)

    def __mStepInducingPoints(self, maxNIter, tol, lr, verbose, nIterDisplay):
        x = self.__kernelMatricesStore.getZ()
        def evalFunc():
            self.__kernelMatricesStore.buildKernelMatrices()
            answer = self.__lowerBound.eval()
            return answer
        updateModelFunc = self.__kernelMatricesStore.buildKernelMatrices
        return self.__setupAndMaximizeStep(x=x, evalFunc=evalFunc, updateModelFunc=updateModelFunc, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose, nIterDisplay=nIterDisplay)

    def __setupAndMaximizeStep(self, x, evalFunc, maxNIter, tol, lr, verbose, nIterDisplay, displayFmt="Iteration: %d, negative lower bound: %f", updateModelFunc=None):
        # print("Start __setupAndMaximizeStep Value: %f"%-evalFunc())
        # pdb.set_trace()
        for i in range(len(x)):
            x[i].requires_grad = True 
        optimizer = torch.optim.Adam(x, lr=lr)
        maxRes = self.__maximizeStep(evalFunc=evalFunc, updateModelFunc=updateModelFunc, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose, nIterDisplay=nIterDisplay, displayFmt=displayFmt)
        for i in range(len(x)):
            x[i].requires_grad = False
        # print("End __setupAndMaximizeStep Value: %f"%-evalFunc())
        # pdb.set_trace()
        return maxRes

    def __maximizeStep(self, evalFunc, updateModelFunc, optimizer, maxNIter,
            tol, verbose, nIterDisplay, displayFmt="Iteration: %d, negative lower bound: %f"):
        prevEval = -float("inf")
        converged = False
        iterCount = 0
        lowerBoundHist = []
        # print('qMu[0]='+str(self.__lowerBound._SparseVariationalLowerBound__klDiv._KLDivergence__inducingPointsPrior._InducingPointsPrior__qMu[0]))
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
        # print('qMu[0]='+str(self.__lowerBound._SparseVariationalLowerBound__klDiv._KLDivergence__inducingPointsPrior._InducingPointsPrior__qMu[0]))
        # pdb.set_trace()
        return {"lowerBound": -curEval, "lowerBoundHist": lowerBoundHist, "converged": converged}
