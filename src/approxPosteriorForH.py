
import pdb
from abc import ABC, abstractmethod
import torch
from kernelMatricesStore import KernelMatricesStore

class ApproxPosteriorForH(ABC):

    def __init__(self, C, d, inducingPointsPrior, kernelMatricesStore):
        self._C = C
        self._d = d
        self._inducingPointsPrior = inducingPointsPrior 
        self._kernelMatricesStore = kernelMatricesStore

    def getApproxPosteriorForHParams(self):
        return self._inducingPointsPrior.getParams()

    def getModelParams(self):
        return [self._C, self._d]
    
    @abstractmethod
    def getMeansAndVariances(self, kFactors=None):
        pass

    @abstractmethod
    def buildKFactors(self):
        pass

class ApproxPosteriorForHForAllNeuronsAllTimes(ApproxPosteriorForH):

    def __init__(self, C, d, inducingPointsPrior, kernelMatricesStore):
        super().__init__(C=C, d=d, inducingPointsPrior=inducingPointsPrior, kernelMatricesStore=kernelMatricesStore)

    def predict(self, testTimes):
        kernels = self._kernelMatricesStore.getKernels()
        Z = self._kernelMatricesStore.getZ()
        Y = self._kernelMatricesStore.getY()
        nTrials = Z[0].shape[0]
        # test times should be a torch tensor
        # test times \in nTestTimes but should be in nTrials x nTestTimes
        testTimesReformatted = torch.matmul(torch.ones(nTrials, dtype=torch.double).reshape(-1, 1), testTimes.reshape(1,-1))
        testTimesReformatted = testTimesReformatted.unsqueeze(2)
        kernelMatricesStore = KernelMatricesStore(kernels=kernels, Z=Z, t=testTimesReformatted, Y=Y)
        Kzz=kernelMatricesStore.getKzz()
        Kzzi=kernelMatricesStore.getKzzi()
        Ktz=kernelMatricesStore.getKtz_allNeuronsAllTimes() 
        Ktt=kernelMatricesStore.getKtt_allNeuronsAllTimes() 
        kFactors = self.__buildKFactorsGivenKernelMatrices(Kzz=Kzz, Kzzi=Kzzi, Ktz=Ktz, Ktt=Ktt)
        qKMu = kFactors["qKMu"]
        qKVar = kFactors["qKVar"]
        qHMu, qHVar = self.__getMeansAndVariancesGivenKFactors(qKMu=qKMu, qKVar=qKVar)
        return qHMu, qHVar, qKMu, qKVar

    def buildKFactors(self):
        Kzz = self._kernelMatricesStore.getKzz()
        Kzzi = self._kernelMatricesStore.getKzzi()
        Ktz = self._kernelMatricesStore.getKtz_allNeuronsAllTimes()
        Ktt = self._kernelMatricesStore.getKtt_allNeuronsAllTimes()
        return self.__buildKFactorsGivenKernelMatrices(Kzz=Kzz, Kzzi=Kzzi, Ktz=Ktz, Ktt=Ktt)

    def __buildKFactorsGivenKernelMatrices(self, Kzz, Kzzi, Ktz, Ktt):
        nTrials = Ktt.shape[0]
        nQuad = Ktt.shape[1]
        nLatent = Ktt.shape[2]

        qKMu = torch.empty((nTrials, nQuad, nLatent), dtype=torch.double)
        qKVar = torch.empty((nTrials, nQuad, nLatent), dtype=torch.double)

        qSigma = self._inducingPointsPrior.buildQSigma()
        for k in range(len(self._inducingPointsPrior.getQMu())):
            # Ak \in nTrials x nInd[k] x 1 
            Ak = torch.matmul(Kzzi[k], self._inducingPointsPrior.getQMu()[k]) 
            qKMu[:,:,k] = torch.squeeze(torch.matmul(Ktz[k], Ak))

            # Bkf \in nTrials x nInd[k] x nQuad
            Bkf = torch.matmul(Kzzi[k], Ktz[k].transpose(dim0=1, dim1=2))
            # mm1f \in nTrials x nInd[k] x nQuad
            mm1f = torch.matmul(qSigma[k]-Kzz[k], Bkf)
            # aux1 \in nTrials x nInd[k] x nQuad
            aux1 = Bkf*mm1f
            # aux2 \in nTrials x nQuad
            aux2 = torch.sum(input=aux1, dim=1)
            # aux3 \in nTrials x nQuad
            aux3 = Ktt[:,:,k]+aux2
            # qKVar \in nTrials x nQuad x nLatent
            qKVar[:,:,k] = aux3
        return {"qKMu": qKMu, "qKVar": qKVar}

    def getMeansAndVariances(self, kFactors=None):
        if kFactors is None:
            kFactors = self.buildKFactors()
        qKMu = kFactors['qKMu']
        qKVar = kFactors['qKVar']
        qHMu, qHVar = self.__getMeansAndVariancesGivenKFactors(qKMu=qKMu, qKVar=qKVar)
        return qHMu, qHVar

    def __getMeansAndVariancesGivenKFactors(self, qKMu, qKVar):
        qHMu = torch.matmul(qKMu, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        qHVar = torch.matmul(qKVar, (torch.t(self._C))**2)
        return qHMu, qHVar

class ApproxPosteriorForHForAllNeuronsAssociatedTimes(ApproxPosteriorForH):

    def __init__(self, C, d, inducingPointsPrior, kernelMatricesStore, neuronForSpikeIndex):
        super().__init__(C=C, d=d, inducingPointsPrior=inducingPointsPrior, kernelMatricesStore=kernelMatricesStore)
        self.__neuronForSpikeIndex = neuronForSpikeIndex

    def buildKFactors(self):
        nTrials = len(self._kernelMatricesStore.getKtt_allNeuronsAssociatedTimes()[0])
        nLatent = len(self._inducingPointsPrior.getQMu())
        # Ak[k] \in nTrial x nInd[k] x 1
        Ak = [torch.matmul(self._kernelMatricesStore.getKzzi()[k], self._inducingPointsPrior.getQMu()[k]) for k in range(nLatent)]
        qSigma = self._inducingPointsPrior.buildQSigma()
        qKMu = [[None] for tr in range(nTrials)]
        qKVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            nSpikesForTrial = self._kernelMatricesStore.getKtt_allNeuronsAssociatedTimes()[0][trialIndex].shape[0]
            # qKMu[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
            qKMu[trialIndex] = torch.empty((nSpikesForTrial, nLatent), dtype=torch.double)
            qKVar[trialIndex] = torch.empty((nSpikesForTrial, nLatent), dtype=torch.double)
            for k in range(nLatent):
                qKMu[trialIndex][:,k] = torch.squeeze(torch.mm(input=self._kernelMatricesStore.getKtz_allNeuronsAssociatedTimes()[k][trialIndex], mat2=Ak[k][trialIndex,:,:]))
                # Bfk \in nInd[k] x nSpikesForTrial[trialIndex]
                Bfk = torch.matmul(self._kernelMatricesStore.getKzzi()[k][trialIndex,:,:], self._kernelMatricesStore.getKtz_allNeuronsAssociatedTimes()[k][trialIndex].transpose(dim0=0, dim1=1))
                # mm1f \in nInd[k] x nSpikesForTrial[trialIndex]
                mm1f = torch.matmul(qSigma[k][trialIndex,:,:]-self._kernelMatricesStore.getKzz()[k][trialIndex,:,:], Bfk)
                # qKVar[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
                qKVar[trialIndex][:,k] = torch.squeeze(self._kernelMatricesStore.getKtt_allNeuronsAssociatedTimes()[k][trialIndex])+torch.sum(a=Bfk*mm1f, axis=0)
        
        return {"qKMu": qKMu, "qKVar": qKVar}

    def __getMeanAndVarianceGivenKFactors(self, qKMu, qKVar):
        qHMu = torch.matmul(qKMu, torch.t(self._C)) + torch.reshape(input=self._d, shape=(1, 1, len(self._d))) # using broadcasting
        qHVar = torch.matmul(qKVar, (torch.t(self._C))**2)
        return qHMu, qHVar

    def getMeansAndVariances(self, kFactors=None):
        if kFactors is None:
            kFactors = self.buildKFactors()
        qKMu = kFactors['qKMu']
        qKVar = kFactors['qKVar']
        qHMu, qHVar = self.__getMeansAndVariancesGivenKFactors(qKMu=qKMu, qKVar=qKVar)
        return qHMu, qHVar

    def __getMeansAndVariancesGivenKFactors(self, qKMu, qKVar):
        nTrials = len(self._kernelMatricesStore.getKtt_allNeuronsAssociatedTimes()[0])
        qHMu = [[None] for tr in range(nTrials)]
        qHVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            qHMu[trialIndex] = torch.sum(qKMu[trialIndex]*self._C[(self.__neuronForSpikeIndex[trialIndex]-1).tolist(),:],dim=1)+self._d[(self.__neuronForSpikeIndex[trialIndex]-1).tolist()]
            qHVar[trialIndex] = torch.sum(qKVar[trialIndex]*(self._C[(self.__neuronForSpikeIndex[trialIndex]-1).tolist(),:])**2,dim=1)
        return qHMu, qHVar

