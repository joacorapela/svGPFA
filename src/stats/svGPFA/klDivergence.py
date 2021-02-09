
import pdb
import torch

class KLDivergence:

    def __init__(self, indPointsLocsKMS, svPosteriorOnIndPoints):
        super(KLDivergence, self).__init__()
        self._indPointsLocsKMS = indPointsLocsKMS
        self._svPosteriorOnIndPoints = svPosteriorOnIndPoints

    def get_svPosteriorOnIndPoints(self):
        return self._svPosteriorOnIndPoints

    def evalSumAcrossLatentsAndTrials(self):
        klDiv = 0
        qSigma = self._svPosteriorOnIndPoints.buildQSigma()
        for k in range(len(self._indPointsLocsKMS.getKzzChol())):
            klDivK = self._evalSumAcrossTrials(
                Kzz=self._indPointsLocsKMS.getKzz()[k],
                KzzChol=self._indPointsLocsKMS.getKzzChol()[k],
                qMu=self._svPosteriorOnIndPoints.getQMu()[k],
                qSigma=qSigma[k])
            klDiv += klDivK
        return klDiv

    def _evalSumAcrossTrials(self, Kzz, KzzChol, qMu, qSigma):
        # ESS \in nTrials x nInd x nInd
        ESS = qSigma + torch.matmul(qMu, qMu.permute(0,2,1))
        nTrials = qMu.shape[0]
        answer = 0
        for trial in range(nTrials):
            _, logdetKzz = Kzz[trial,:,:].slogdet() # O(n^3)
            _, logdetQSigma = qSigma[trial,:,:].slogdet() # O(n^3)
            traceTerm = torch.trace(
                torch.cholesky_solve(ESS[trial,:,:], KzzChol[trial,:,:]))
            trialKL = .5*(traceTerm+logdetKzz-logdetQSigma-ESS.shape[1])
            answer += trialKL
        return answer
