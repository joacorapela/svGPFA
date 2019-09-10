
import pdb
import torch

class SparseVariationalEM:

    def __init__(self, lowerBound):
        self.__lowerBound = lowerBound

    def maximize(self, emMaxNIter=100, eStepMaxNIter=1000, eStepTol=1e-3, eStepLR=1e-3, mStepModelParamsMaxNIter=1000, mStepModelParamsTol=1e-3, mStepModelParamsLR=1e-3, verbose=False):
        '''
        m0, SVec0, SDiag0 \in nInd x nNeruons x nTrials
        '''
        iter = 0
        while iter<emMaxNIter:
            rc = self.__eStep(maxNIter=eStepMaxNIter, tol=eStepTol, lr=eStepLR, verbose=verbose)
            rc = self.__mStepModelParams(maxNIter=mStepModelParamsMaxNIter, tol=mStepModelParamsTol, lr=mStepLModelParamsR, verbose=verbose)
            
            
    def __eStep(self, maxNIter, tol, lr, verbose):
        x = self.__lowerBound.getApproxPosteriorForHParams()
        return self.__setupAndMaximizeStep(x=x, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose)

    def __mStepModelParams(self, maxNIter, tol, lr, verbose):
        x = self.__lowerBound.getModelParams()
        return self.__setupAndMaximizeStep(x=x, maxNIter=maxNIter, tol=tol, lr=lr, verbose=verbose)

    def __setupAndMaximizeStep(self, x, maxNIter, tol, lr, verbose):
        for i in range(len(x)):
            x[i].requires_grad = True 
        optimizer = torch.optim.Adam(x, lr=lr)
        # optimizer = torch.optim.SGD(x, lr=lr, momentum=0.9)
        maxRes = self.__maximizeStep(x=x, optimizer=optimizer, maxNIter=maxNIter, tol=tol, verbose=verbose)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def __maximizeStep(self, x, optimizer, maxNIter, tol, verbose, nIterDisplay=1):
        prevEval = -float("inf")
        converged = False
        iterCount = 0
        lowerBoundHist = []
        while not converged and iterCount<maxNIter:
            iterCount += 1
            optimizer.zero_grad()
            # pdb.set_trace()
            curEval = -self.__lowerBound.eval()
            if verbose and iterCount%nIterDisplay==0:
                print("Iteration: %d, Value: %.04f"%(iterCount, curEval))
            lowerBoundHist.append(-curEval)
            if curEval<prevEval and prevEval-curEval<tol:
                converged = True
            else:
                curEval.backward(retain_graph=True)
                optimizer.step()
        return {"lowerBound": -curEval, "lowerBoundHist": lowerBoundHist, "converged": converged}
