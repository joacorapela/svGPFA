
import pdb
from abc import ABC, abstractmethod
import torch

class ApproxPosteriorForH(ABC):

    def __init__(self, C, d, inducingPointsPrior, covMatricesStore):
        self._C = C
        self._d = d
        self._inducingPointsPrior = inducingPointsPrior 
        self._covMatricesStore = covMatricesStore

    def getApproxPosteriorForHParams(self):
        return self._inducingPointsPrior.getParams()

    def getModelParams(self):
        # return [self._C]
        # return [self._d]
        return [self._C, self._d]
                
    def getMeanAndVarianceAtQuadPoints(self):

        nTrials = self._covMatricesStore.getQuadKtt().shape[0]
        nQuad = self._covMatricesStore.getQuadKtt().shape[1]
        nLatent = self._covMatricesStore.getQuadKtt().shape[2]

        muK = torch.empty((nTrials, nQuad, nLatent), dtype=torch.double)
        varK = torch.empty((nTrials, nQuad, nLatent), dtype=torch.double)

        qSigma = self._inducingPointsPrior.buildQSigma()
        for k in range(len(self._inducingPointsPrior.getQMu())):
            # Ak \in nTrials x nInd[k] x 1 
            Ak = torch.matmul(self._covMatricesStore.getKzzi()[k], self._inducingPointsPrior.getQMu()[k]) 
            muK[:,:,k] = torch.squeeze(torch.matmul(self._covMatricesStore.getQuadKtz()[k], Ak))
            # Bkf \in nTrials x nInd[k] x nQuad
            Bkf = torch.matmul(self._covMatricesStore.getKzzi()[k], self._covMatricesStore.getQuadKtz()[k].transpose(dim0=1, dim1=2))
            # mm1f \in nTrials x nInd[k] x nQuad
            mm1f = torch.matmul(qSigma[k]-self._covMatricesStore.getKzz()[k], Bkf)

            # aux1 \in nTrials x nInd[k] x nQuad
            aux1 = Bkf*mm1f
            # aux2 \in nTrials x nQuad
            aux2 = torch.sum(input=aux1, dim=1)
            # aux3 \in nTrials x nQuad
            aux3 = self._covMatricesStore.getQuadKtt()[:,:,k]+aux2
            # varK \in nTrials x nQuad x nLatent
            varK[:,:,k] = aux3

        qHMu = torch.matmul(muK, torch.t(self._C)) + \
                torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        qHVar = torch.matmul(varK, (torch.t(self._C))**2)
        return (qHMu, qHVar)

class PointProcessApproxPosteriorForH(ApproxPosteriorForH):

    def __init__(self, C, d, inducingPointsPrior, covMatricesStore, neuronForSpikeIndex):
        super().__init__(C=C, d=d, inducingPointsPrior=inducingPointsPrior, covMatricesStore=covMatricesStore)
        self.__neuronForSpikeIndex = neuronForSpikeIndex

    def getMeanAndVarianceAtSpikeTimes(self):

        nTrials = len(self._covMatricesStore.getSpikeKtt()[0])
        nLatent = len(self._inducingPointsPrior.getQMu())
        # Ak[k] \in nTrial x nInd[k] x 1
        Ak = [torch.matmul(self._covMatricesStore.getKzzi()[k], self._inducingPointsPrior.getQMu()[k]) for k in range(nLatent)]
        qSigma = self._inducingPointsPrior.buildQSigma()
        qKMu = [None] * nTrials
        qKVar = [None] * nTrials
        for trialIndex in range(nTrials):
            nSpikesForTrial = self._covMatricesStore.getSpikeKtt()[0][trialIndex].shape[0]
            # qKMu[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
            qKMu[trialIndex] = torch.empty((nSpikesForTrial, nLatent), dtype=torch.double)
            qKVar[trialIndex] = torch.empty((nSpikesForTrial, nLatent), dtype=torch.double)
            for k in range(nLatent):
                qKMu[trialIndex][:,k] = torch.squeeze(torch.mm(input=self._covMatricesStore.getSpikeKtz()[k][trialIndex], mat2=Ak[k][trialIndex,:,:]))
                # Bfk \in nInd[k] x nSpikesForTrial[trialIndex]
                Bfk = torch.matmul(self._covMatricesStore.getKzzi()[k][trialIndex,:,:], self._covMatricesStore.getSpikeKtz()[k][trialIndex].transpose(dim0=0, dim1=1))
                # mm1f \in nInd[k] x nSpikesForTrial[trialIndex]
                mm1f = torch.matmul(qSigma[k][trialIndex,:,:]-self._covMatricesStore.getKzz()[k][trialIndex,:,:], Bfk)
                # qKVar[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
                qKVar[trialIndex][:,k] = torch.squeeze(self._covMatricesStore.getSpikeKtt()[k][trialIndex])+torch.sum(a=Bfk*mm1f, axis=0)
        qHMu = [None] * nTrials
        qHVar = [None] * nTrials
        for trialIndex in range(nTrials):
            qHMu[trialIndex] = torch.sum(qKMu[trialIndex]*self._C[(self.__neuronForSpikeIndex[trialIndex]-1).tolist(),:],dim=1)+self._d[(self.__neuronForSpikeIndex[trialIndex]-1).tolist()]
            qHVar[trialIndex] = torch.sum(qKVar[trialIndex]*(self._C[(self.__neuronForSpikeIndex[trialIndex]-1).tolist(),:])**2,dim=1)
        return qHMu, qHVar
