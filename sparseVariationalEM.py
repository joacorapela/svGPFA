
import pdb
import torch

class SparseVariationalEM:

    def __init__(self, lowerBound):
        self.__lowerBound = lowerBound

    def maximize(self, emMaxNIter=100, eStepMaxNIter=1000, eStepTol=1e-3, eStepLR=1e-3):
        '''
        m0, SVec0, SDiag0 \in nInd x nNeruons x nTrials
        '''
        iter = 0
        while iter<emMaxNIter:
            rc = self.__eStep(maxNIter=eStepMaxNIter, tol=eStepTol, lr=eStepLR)
            
    def __eStep(self, maxNIter, tol, lr):
        x = self.__lowerBound.getApproxPosteriorForHParams()
        for i in range(len(x)):
            x[i].requires_grad = True 
        optimizer = torch.optim.Adam(x, lr=lr)
        maxRes = self.__maximizeStep(x=x, optimizer=optimizer, maxNIter=maxNIter, tol=tol)
        for i in range(len(x)):
            x[i].requires_grad = False
        return maxRes

    def __maximizeStep(self, x, optimizer, maxNIter, tol):
        prevEval = -float("inf")
        converged = False
        iterCount = 0
        lowerBoundHist = []
        while not converged and iterCount<maxNIter:
            iterCount += 1
            optimizer.zero_grad()
            curEval = -self.__lowerBound.eval()
            lowerBoundHist.append(-curEval)
            if curEval<prevEval and prevEval-curEval<tol:
                converged = True
            else:
                curEval.backward(retain_graph=True)
                optimizer.step()
        pdb.set_trace()
        return {"lowerBound": curEval, "lowerBoundHist": lowerBoundHist, "converged": converged}
