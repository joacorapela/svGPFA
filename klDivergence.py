
import pdb
import torch

class KLDivergence:

    def __init__(self, Kzzi, inducingPointsPrior):
        self.__Kzzi = Kzzi
        self.__inducingPointsPrior = inducingPointsPrior

    def evalSumAcrossLatentsAndTrials(self):
        klDiv = 0
        qSigma = self.__inducingPointsPrior.buildQSigma()
        for k in range(len(self.__Kzzi)):
            klDivK = self.__evalSumAcrossTrials(Kzzi=self.__Kzzi[k], qMu=self.__inducingPointsPrior.getQMu()[k], qSigma=qSigma[k])
            klDiv += klDivK
        return klDiv

    def __evalSumAcrossTrials(self, Kzzi, qMu, qSigma):
        # ESS \in nTrials x nInd x nInd
        ESS = qSigma + torch.matmul(qMu, qMu.permute(0,2,1))
        nTrials = qMu.shape[0]
        answer = 0
        for trial in range(nTrials):
            _, logdetKzzi = Kzzi[trial,:,:].slogdet()
            _, logdetQSigma = qSigma[trial,:,:].slogdet()
            # aux3 = np.dot(np.ndarray.flatten(Kzzi[trial,:,:]), np.ndarray.flatten(ESS[trial,:,:]))
            # aux4 = ESS.shape[1]
            trialKL = .5*(-logdetKzzi-logdetQSigma+torch.flatten(Kzzi[trial,:,:]).dot(torch.flatten(ESS[trial,:,:]))-ESS.shape[1])
            answer += trialKL
            # print("aux1=%f\naux2=%f\naux3=%f\naux4=%f\nkldiv_nn=%f,kldiv=%f"%(logdetKzzi, logdetQSigma, aux3, aux4, trialKL, answer))
            # pdb.set_trace()
        return answer
