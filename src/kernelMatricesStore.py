
import pdb
import torch
from utils import pinv3D

class KernelMatricesStore:

    def __init__(self, kernels, Z, t, Y):
        # Z[k] \in nTrials x nInd[k] x 1
        self.__kernels = kernels
        self.__Z = Z
        self.__t = t
        self.__Y = Y
        self.buildKernelMatrices()

    def buildKernelMatrices(self):
        self.__buildZKernelMatrices()
        self.__buildZTKernelMatrices_allNeuronsAllTimes(t=self.__t)
        self.__buildZTKernelMatrices_allNeuronsAssociatedTimes(Y=self.__Y)

    def getKernels(self):
        # this method should not exist. It is here only for a temporary patch.
        return self.__kernels

    def getZ(self):
        return self.__Z

    def getY(self):
        # this method should not exist. It is here only for a temporary patch.
        return self.__Y

    def getKernelsVariableParameters(self):
        answer = []
        for i in range(len(self.__kernels)):
            answer.append(self.__kernels[i].getVariableParameters())
        return answer

    def getKzz(self):
        return self._Kzz

    def getKzzi(self):
        return self._Kzzi

    def getKtz_allNeuronsAllTimes(self):
        return self._Ktz_allNeuronsAllTimes

    def getKtt_allNeuronsAllTimes(self):
        return self._Ktt_allNeuronsAllTimes

    def getKtz_allNeuronsAssociatedTimes(self):
        return self._Ktz_allNeuronsAssociatedTimes

    def getKtt_allNeuronsAssociatedTimes(self):
        return self._Ktt_allNeuronsAssociatedTimes

    def __buildZKernelMatrices(self, epsilon=1e-5):
        nLatent = len(self.__kernels)
        self._Kzz = [[None] for k in range(nLatent)]
        self._Kzzi = [[None] for k in range(nLatent)]

        for k in range(nLatent):
            self._Kzz[k] = self.__kernels[k].buildKernelMatrix(X1=self.__Z[k])+epsilon*torch.eye(n=self.__Z[k].shape[1], dtype=torch.double)
            self._Kzzi[k] = pinv3D(self._Kzz[k])

    def __buildZTKernelMatrices_allNeuronsAllTimes(self, t):
        # t \in nTrials x nQuad x 1
        nLatent = len(self.__Z)
        self._Ktz_allNeuronsAllTimes = [[None] for k in range(nLatent)]
        self._Ktt_allNeuronsAllTimes = torch.zeros(t.shape[0], t.shape[1], nLatent, dtype=torch.double)
        for k in range(nLatent):
            self._Ktz_allNeuronsAllTimes[k] = self.__kernels[k].buildKernelMatrix(X1=t, X2=self.__Z[k])
            self._Ktt_allNeuronsAllTimes[:,:,k] = self.__kernels[k].buildKernelMatrixDiag(X=t).squeeze()

    def __buildZTKernelMatrices_allNeuronsAssociatedTimes(self, Y):
        # Y[tr] \in nSpikes of all neurons in trial tr
        nLatent = len(self.__Z)
        nTrial = self.__Z[0].shape[0]
        self._Ktz_allNeuronsAssociatedTimes = [[[None] for tr in range(nTrial)] for k in range(nLatent)]
        self._Ktt_allNeuronsAssociatedTimes = [[[None] for tr in  range(nTrial)] for k in range(nLatent)]

        for k in range(nLatent):
            for tr in range(nTrial):
                self._Ktz_allNeuronsAssociatedTimes[k][tr] = self.__kernels[k].buildKernelMatrix(X1=Y[tr], X2=self.__Z[k][tr,:,:])
                self._Ktt_allNeuronsAssociatedTimes[k][tr] = self.__kernels[k].buildKernelMatrixDiag(X=Y[tr])
