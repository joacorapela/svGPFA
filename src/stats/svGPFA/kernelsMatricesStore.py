
import pdb
import torch
from abc import ABC, abstractmethod
import utils.svGPFA.miscUtils

class KernelsMatricesStore(ABC):

    @abstractmethod
    def buildKernelsMatrices(self):
        pass

    def setKernels(self, kernels):
        self._kernels = kernels

    def setInitialParams(self, initialParams):
        self.setIndPointsLocs(indPointsLocs=initialParams["inducingPointsLocs0"])
        self.setKernelsParams(kernelsParams=initialParams["kernelsParams0"])

    def setKernelsParams(self, kernelsParams):
        for k in range(len(self._kernels)):
            self._kernels[k].setParams(kernelsParams[k])

    def setIndPointsLocs(self, indPointsLocs):
        self._indPointsLocs = indPointsLocs

    def getIndPointsLocs(self):
        return self._indPointsLocs

    def getKernels(self):
        return self._kernels

    def getKernelsParams(self):
        answer = []
        for i in range(len(self._kernels)):
            answer.append(self._kernels[i].getParams())
        return answer

class IndPointsLocsKMS(KernelsMatricesStore):

    def setEpsilon(self, epsilon):
        self._epsilon = epsilon

    def buildKernelsMatrices(self):
        nLatent = len(self._kernels)
        self._Kzz = [[None] for k in range(nLatent)]
        self._KzzChol = [[None] for k in range(nLatent)]

        for k in range(nLatent):
            self._Kzz[k] = (self._kernels[k].buildKernelMatrix(X1=self._indPointsLocs[k])+self._epsilon*torch.eye(n=self._indPointsLocs[k].shape[1], dtype=self._indPointsLocs[k].dtype, device=self._indPointsLocs[k].device))
            # self._Kzz[k] =
            # self._kernels[k].buildKernelMatrix(X1=self._indPointsLocs[k])
            self._KzzChol[k] = utils.svGPFA.miscUtils.chol3D(self._Kzz[k]) # O(n^3)

    def getKzz(self):
        return self._Kzz

    def getKzzChol(self):
        return self._KzzChol

    def getEpsilon(self):
        return self._epsilon

class IndPointsLocsAndTimesKMS(KernelsMatricesStore):

    def setTimes(self, times):
        self._t = times

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

    def getKttDiag(self):
        return self._KttDiag

class IndPointsLocsAndAllTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        # t \in nTrials x nQuad x 1
        nLatent = len(self._indPointsLocs)
        self._Ktz = [[None] for k in range(nLatent)]
        self._KttDiag = torch.zeros(self._t.shape[0], self._t.shape[1], nLatent,
                                    dtype=self._t.dtype, device=self._t.device)
        for k in range(nLatent):
            self._Ktz[k] = self._kernels[k].buildKernelMatrix(X1=self._t,
                                                              X2=self._indPointsLocs[k])
            self._KttDiag[:,:,k] = self._kernels[k].buildKernelMatrixDiag(X=self._t).squeeze()

    def buildKttKernelsMatrices(self):
        # t \in nTrials x nQuad x 1
        nLatent = len(self._indPointsLocs)
        self._Ktt = [[None] for k in range(nLatent)]

        for k in range(nLatent):
            self._Ktt[k] = self._kernels[k].buildKernelMatrix(X1=self._t, X2=self._t)

class IndPointsLocsAndAssocTimesKMS(IndPointsLocsAndTimesKMS):

    def buildKernelsMatrices(self):
        nLatent = len(self._indPointsLocs)
        nTrial = self._indPointsLocs[0].shape[0]
        self._Ktz = [[[None] for tr in range(nTrial)] for k in range(nLatent)]
        self._KttDiag = [[[None] for tr in  range(nTrial)] for k in range(nLatent)]

        for k in range(nLatent):
            for tr in range(nTrial):
                self._Ktz[k][tr] = self._kernels[k].buildKernelMatrix(X1=self._t[tr], X2=self._indPointsLocs[k][tr,:,:])
                self._KttDiag[k][tr] = self._kernels[k].buildKernelMatrixDiag(X=self._t[tr])

