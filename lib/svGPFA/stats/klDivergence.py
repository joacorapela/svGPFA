
import pdb
import torch

class KLDivergence:

    def __init__(self, indPointsLocsKMS, svPosteriorOnIndPoints):
        super(KLDivergence, self).__init__()
        self._indPointsLocsKMS = indPointsLocsKMS
        self._svPosteriorOnIndPoints = svPosteriorOnIndPoints

    def get_indPointsLocsKMS(self):
        return self._indPointsLocsKMS

    def get_svPosteriorOnIndPoints(self):
        return self._svPosteriorOnIndPoints

    def evalSumAcrossLatentsAndTrials(self):
        klDiv = 0
        qSigma = self._svPosteriorOnIndPoints.buildCov()
        nLatents = len(qSigma)
        for k in range(nLatents):
            klDivK = self._evalSumAcrossTrials(
                Kzz=self._indPointsLocsKMS.getKzz()[k],
                qMu=self._svPosteriorOnIndPoints.getMean()[k],
                qSigma=qSigma[k],
                latentIndex=k)
            klDiv += klDivK
        return klDiv

    def _evalSumAcrossTrials(self, Kzz, qMu, qSigma, latentIndex):
        # ESS \in nTrials x nInd x nInd
        ESS = qSigma + torch.matmul(qMu, qMu.permute(0,2,1))
        nTrials = qMu.shape[0]
        answer = 0
        for trialIndex in range(nTrials):
            _, logdetKzz = Kzz[trialIndex,:,:].slogdet() # O(n^3)
            _, logdetQSigma = qSigma[trialIndex,:,:].slogdet() # O(n^3)
            # traceTerm = torch.trace(torch.cholesky_solve(ESS[trialIndex,:,:],
            # KzzInv[trialIndex,:,:]))
            traceTerm = torch.trace(self._indPointsLocsKMS.solveForLatentAndTrial(ESS[trialIndex,:,:], latentIndex=latentIndex, trialIndex=trialIndex))
            trialKL = .5*(traceTerm+logdetKzz-logdetQSigma-ESS.shape[1])
            answer += trialKL
        return answer
