
import pdb
from abc import ABC, abstractmethod
import math
import torch

class Kernel(ABC):

    @abstractmethod
    def buildKernelMatrix(self, X1, X2=None):
        pass

    @abstractmethod
    def buildKernelMatrixDiag(self, X):
        pass

    def getParams(self):
        params = self._params[self._freeParams]
        return params

    def setParams(self, params):
        self._params[self._freeParams] = params

'''
class AddDiagKernel(Kernel):
    def __init__(self, kernel, epsilon=1e-5):
        self.__kernel = kernel
        self.__epsilon = epsilon

    def buildKernelMatrix(self, X1, X2=None):
        covMatrix = self.__kernel.buildKernelMatrix(X1=X1, X2=X2)
        covMatrixPlusDiag = (covMatrix + 
                             self.__epsilon*torch.eye(n=covMatrix.shape[0], 
                                                      dtype=torch.double))
        return covMatrixPlusDiag

    def buildKernelMatrixDiag(self, X):
        return self.__kernel.buildKernelMatrixDiag(X=X)

    def setParams(self, params):
        self.__kernel.setParams(params)

    def getParams(self):
        params = self.__kernel.getParams()
        return params
'''

class ExponentialQuadraticKernel(Kernel):

    def __init__(self, scale=None, lengthScale=None):
        self._params = [None, None]
        self._freeParams = [True, True]
        if scale is not None:
            self._params[0] = scale
            self._freeParams[0] = False
        if lengthScale is not None:
            self._params[1] = lengthScale
            self._freeParams[1]=False

    def buildKernelMatrix(self, X1, X2=None):
        scale = self._params[0]
        lengthScale = self._params[1]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self._params[0]
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

class PeriodicKernel(Kernel):
    def __init__(self, scale=None, lengthScale=None, period=None):
        self._params = [None, None, None]
        self._freeParams = [True, True, True]
        if scale is not None:
            self._params[0] = scale
            self._freeParams[0] = False
        if lengthScale is not None:
            self._params[1] = lengthScale
            self._freeParams[1] = False
        if period is not None:
            self._params[2] = period
            self._freeParams[2] = False

    def buildKernelMatrix(self, X1, X2=None):
        scale = self._params[0]
        lengthScale = self._params[1]
        period = self._params[2]
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            sDistance = X1-X2.transpose(1, 2)
        else:
            sDistance = X1.reshape(-1,1)-X2.reshape(1,-1)
        rr = math.pi*sDistance/period
        covMatrix = scale**2*torch.exp(-2*torch.sin(rr)**2/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale = self._params[0]
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=torch.double)
        return covMatrixDiag

