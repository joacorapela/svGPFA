
import pdb
from abc import ABC, abstractmethod
import torch
import scipy.stats
from .kernelMatricesStore import IndPointsLocsAndAllTimesKMS

class SVPosteriorOnLatents(ABC):

    def __init__(self, svPosteriorOnIndPoints, indPointsLocsKMS,
                 indPointsLocsAndTimesKMS):
        self._svPosteriorOnIndPoints = svPosteriorOnIndPoints
        self._indPointsLocsKMS = indPointsLocsKMS
        self._indPointsLocsAndTimesKMS=indPointsLocsAndTimesKMS

    @abstractmethod
    def computeMeansAndVars(self):
        pass

    @abstractmethod
    def buildKernelsMatrices(self):
        pass

    @abstractmethod
    def setKernels(self):
        pass

    @abstractmethod
    def setInitialParams(self, initialParams):
        pass

    def setTimes(self, times):
        self._indPointsLocsAndTimesKMS.setTimes(times=times)

    @abstractmethod
    def setIndPointsLocs(self, indPointsLocs):
        pass

    def getSVPosteriorOnIndPointsParams(self):
        return self._svPosteriorOnIndPoints.getParams()

    def getKernelsParams(self):
        return self._indPointsLocsKMS.getKernelsParams()

    def getIndPointsLocs(self):
        return self._indPointsLocsKMS.getIndPointsLocs()

    def setIndPointsLocsKMSEpsilon(self, indPointsLocsKMSEpsilon):
        self._indPointsLocsKMS.setEpsilon(epsilon=indPointsLocsKMSEpsilon)

class SVPosteriorOnLatentsAllTimes(SVPosteriorOnLatents):

    def predict(self, newTimes):
        Z = self.getIndPointsLocs()
        kernels = self._indPointsLocsKMS.getKernels()
        nTrials = Z[0].shape[0]

        # test times \in nTestTimes but should be in nTrials x nTestTimes
        newTimesReformatted = torch.matmul(
            torch.ones(nTrials,
                dtype=newTimes.dtype,
                device=newTimes.device).reshape(-1, 1),
            newTimes.reshape(1,-1))
        newTimesReformatted = newTimesReformatted.unsqueeze(2)

        indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
        indPointsLocsAndAllTimesKMS.setTimes(times=newTimesReformatted)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(
            indPointsLocs=self.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()

        svPosteriorOnLatents = SVPosteriorOnLatentsAllTimes(
            svPosteriorOnIndPoints=self._svPosteriorOnIndPoints,
            indPointsLocsKMS=self._indPointsLocsKMS,
            indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
        qKMu, qKVar = svPosteriorOnLatents.computeMeansAndVars()

        return qKMu, qKVar

    def computeMeansAndVars(self):
        Kzz = self._indPointsLocsKMS.getKzz()
        KzzPinv = self._indPointsLocsKMS.getKzzPinv()
        Ktz = self._indPointsLocsAndTimesKMS.getKtz()
        KttDiag = self._indPointsLocsAndTimesKMS.getKttDiag()
        answer = self.__computeMeansAndVarsGivenKernelMatrices(Kzz=Kzz, KzzPinv=KzzPinv, Ktz=Ktz, KttDiag=KttDiag)
        return answer

    def computeMeans(self, times):
        Kzz = self._indPointsLocsKMS.getKzz()
        KzzPinv = self._indPointsLocsKMS.getKzzPinv()

        indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=self._indPointsLocsKMS.getKernels())
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(indPointsLocs=self._indPointsLocsKMS.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.setTimes(times=times)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()

        answer = self.__computeMeansGivenKernelMatrices(Kzz=Kzz, KzzPinv=KzzPinv, Ktz=Ktz)
        return answer

    def sample(self, times, regFactor=1e-3):
        Kzz = self._indPointsLocsKMS.getKzz()
        KzzPinv = self._indPointsLocsKMS.getKzzPinv()

        indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=self._indPointsLocsKMS.getKernels())
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(indPointsLocs=self._indPointsLocsKMS.getIndPointsLocs())
        indPointsLocsAndAllTimesKMS.setTimes(times=times)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        indPointsLocsAndAllTimesKMS.buildKttKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()
        Ktt = indPointsLocsAndAllTimesKMS.getKtt()

        qMu = self._svPosteriorOnIndPoints.getQMu()
        qSigma = self._svPosteriorOnIndPoints.buildQSigma()

        nLatents = len(Kzz)
        nTrials = Kzz[0].shape[0]
        samples = [[] for r in range(nTrials)]
        for r in range(nTrials):
            samples[r] = torch.empty((nLatents, Ktt[0].shape[1]), dtype=Kzz[0].dtype)
            for k in range(nLatents):
                print("Processing trial {:d} and latent {:d}".format(r, k))
                Kzzrk = Kzz[k][r,:,:]
                KzzPinvrk = KzzPinv[k][r,:,:]
                Ktzrk = Ktz[k][r,:,:]
                Kttrk = Ktt[k][r,:,:]
                qMurk = qMu[k][r,:,:]
                qSigmark = qSigma[k][r,:,:]

                ### being compute mean ###
                b = torch.matmul(KzzPinvrk, qMurk)
                meanrk = torch.squeeze(Ktzrk.matmul(b))
                ### end compute mean ###

                ### being compute covar ###
                B = torch.matmul(KzzPinvrk, torch.t(Ktzrk))
                covarrk = Kttrk+torch.t(B).matmul(qSigmark-Kzzrk).matmul(B)
                ### end compute covar ###

                covarrk += torch.eye(covarrk.shape[0])*regFactor
                covarrk = covarrk.detach()
                mn = scipy.stats.multivariate_normal(mean=meanrk, cov=covarrk)
                samples[r][k,:] = torch.from_numpy(mn.rvs())
        return samples

    def __computeMeansAndVarsGivenKernelMatrices(self, Kzz, KzzPinv, Ktz, KttDiag):
        nTrials = KttDiag.shape[0]
        nQuad = KttDiag.shape[1]
        nLatent = KttDiag.shape[2]

        qKMu = torch.empty((nTrials, nQuad, nLatent), dtype=Kzz[0].dtype,
            device=Kzz[0].device)
        qKVar = torch.empty((nTrials, nQuad, nLatent), dtype=Kzz[0].dtype,
            device=Kzz[0].device)

        qSigma = self._svPosteriorOnIndPoints.buildQSigma()
        for k in range(len(self._svPosteriorOnIndPoints.getQMu())):
            # Ak \in nTrials x nInd[k] x 1
            Ak = torch.matmul(KzzPinv[k], self._svPosteriorOnIndPoints.getQMu()[k])
            # qKMu \in  nTrial x nQuad x nLatent
            qKMu[:,:,k] = torch.squeeze(torch.matmul(Ktz[k], Ak))

            # Bkf \in nTrials x nInd[k] x nQuad
            Bkf = torch.matmul(KzzPinv[k], Ktz[k].transpose(dim0=1, dim1=2))
            # mm1f \in nTrials x nInd[k] x nQuad
            mm1f = torch.matmul(qSigma[k]-Kzz[k], Bkf)
            # aux1 \in nTrials x nInd[k] x nQuad
            aux1 = Bkf*mm1f
            # aux2 \in nTrials x nQuad
            aux2 = torch.sum(input=aux1, dim=1)
            # aux3 \in nTrials x nQuad
            aux3 = KttDiag[:,:,k]+aux2
            # qKVar \in nTrials x nQuad x nLatent
            qKVar[:,:,k] = aux3
        return qKMu, qKVar

    def __computeMeansGivenKernelMatrices(self, Kzz, KzzPinv, Ktz):
        nTrials = Ktz[0].shape[0]
        nQuad = Ktz[0].shape[1]
        nLatent = len(Ktz)

        qKMu = torch.empty((nTrials, nQuad, nLatent), dtype=Kzz[0].dtype,
                           device=Kzz[0].device)
        for k in range(len(self._svPosteriorOnIndPoints.getQMu())):
            # Ak \in nTrials x nInd[k] x 1
            Ak = torch.matmul(KzzPinv[k], self._svPosteriorOnIndPoints.getQMu()[k])
            # qKMu \in  nTrial x nQuad x nLatent
            qKMu[:,:,k] = torch.squeeze(torch.matmul(Ktz[k], Ak))
        return qKMu

    def buildKernelsMatrices(self):
        self._indPointsLocsKMS.buildKernelsMatrices()
        self._indPointsLocsAndTimesKMS.buildKernelsMatrices()

    def setKernels(self, kernels):
        self._indPointsLocsKMS.setKernels(kernels=kernels)
        self._indPointsLocsAndTimesKMS.setKernels(kernels=kernels)

    def setInitialParams(self, initialParams):
        self._svPosteriorOnIndPoints.setInitialParams(initialParams=initialParams["svPosteriorOnIndPoints"])
        self._indPointsLocsKMS.setInitialParams(initialParams=initialParams["kernelsMatricesStore"])
        self._indPointsLocsAndTimesKMS.setInitialParams(initialParams=initialParams["kernelsMatricesStore"])

    def setIndPointsLocs(self, indPointsLocs):
        self._indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
        self._indPointsLocsAndTimesKMS.setIndPointsLocs(indPointsLocs=
                                                        indPointsLocs)

class SVPosteriorOnLatentsAssocTimes(SVPosteriorOnLatents):

    def computeMeansAndVars(self):
        Kzz = self._indPointsLocsKMS.getKzz()
        KzzPinv = self._indPointsLocsKMS.getKzzPinv()
        Ktz = self._indPointsLocsAndTimesKMS.getKtz()
        KttDiag = self._indPointsLocsAndTimesKMS.getKttDiag()
        return self.__computeMeansAndVarsGivenKernelMatrices(Kzz=Kzz,
                                                             KzzPinv=KzzPinv,
                                                             Ktz=Ktz,
                                                             KttDiag=KttDiag)

    def __computeMeansAndVarsGivenKernelMatrices(self, Kzz, KzzPinv, Ktz,
                                                 KttDiag):
        nTrials = len(KttDiag[0])
        nLatent = len(self._svPosteriorOnIndPoints.getQMu())
        # Ak[k] \in nTrial x nInd[k] x 1
        Ak = [torch.matmul(KzzPinv[k], self._svPosteriorOnIndPoints.getQMu()[k]) for k in range(nLatent)]
        qSigma = self._svPosteriorOnIndPoints.buildQSigma()
        qKMu = [[None] for tr in range(nTrials)]
        qKVar = [[None] for tr in range(nTrials)]
        for trialIndex in range(nTrials):
            nSpikesForTrial = KttDiag[0][trialIndex].shape[0]
            # qKMu[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
            qKMu[trialIndex] = torch.empty((nSpikesForTrial, nLatent),
                                           dtype=Kzz[0].dtype,
                                           device=Kzz[0].device)
            qKVar[trialIndex] = torch.empty((nSpikesForTrial, nLatent),
                                            dtype=Kzz[0].dtype,
                                            device=Kzz[0].device)
            for k in range(nLatent):
                qKMu[trialIndex][:,k] = torch.squeeze(torch.mm(input=Ktz[k][trialIndex], mat2=Ak[k][trialIndex,:,:]))
                # Bfk \in nInd[k] x nSpikesForTrial[trialIndex]
                Bfk = torch.matmul(
                    KzzPinv[k][trialIndex,:,:],
                    Ktz[k][trialIndex].transpose(dim0=0, dim1=1))
                # mm1f \in nInd[k] x nSpikesForTrial[trialIndex]
                mm1f = torch.matmul(qSigma[k][trialIndex,:,:]-Kzz[k][trialIndex,:,:], Bfk)
                # qKVar[trialIndex] \in nSpikesForTrial[trialIndex] x nLatent
                qKVar[trialIndex][:,k] = torch.squeeze(KttDiag[k][trialIndex])+torch.sum(a=Bfk*mm1f, axis=0)

        return qKMu, qKVar

    def buildKernelsMatrices(self):
        # not asking _indPointsLocsKMS to buildMatrices because
        # SVPosteriorOnLatentsAlTimes already did so
        self._indPointsLocsAndTimesKMS.buildKernelsMatrices()

    def setKernels(self, kernels):
        self._indPointsLocsAndTimesKMS.setKernels(kernels=kernels)

    def setInitialParams(self, initialParams):
        self._indPointsLocsAndTimesKMS.setInitialParams(initialParams=initialParams["kernelsMatricesStore"])

    def setIndPointsLocs(self, indPointsLocs):
        # not asking _indPointsLocsKMS to setIndPointsLocs becasue
        # SVPosteriorOnLatentsAlTimes already did so
        self._indPointsLocsAndTimesKMS.setIndPointsLocs(indPointsLocs=
                                                        indPointsLocs)

