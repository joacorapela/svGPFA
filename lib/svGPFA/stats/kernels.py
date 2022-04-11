
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
        return self._params

    def setParams(self, params):
        self._params = params

    @abstractmethod
    def getNamedParams(self):
        pass

class ExponentialQuadraticKernel(Kernel):

    def __init__(self, scale=1.0, lengthscaleScale=1.0, dtype=torch.double):
        self._scale = torch.tensor(scale, dtype=dtype)
        self._lengthscaleScale = lengthscaleScale

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthscale = self._getAllParams()
        if not hasattr(self, "_lengthscaleScale"):
            self._lengthscaleScale = 1.0
        lengthscale = lengthscale/self._lengthscaleScale

        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthscale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale, lengthscale = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        scale = self._scale
        lengthscale = self._params[0]
        return scale, lengthscale

    def getScaledParams(self):
        scaledParams = torch.tensor([self._params[0]/self._lengthscaleScale])
        return scaledParams

    def getNamedParams(self):
        scale, lengthscale = self._getAllParams()
        answer = {"scale": scale, "lengthscale": lengthscale}
        return answer

class PeriodicKernel(Kernel):
    def __init__(self, scale=1.0, lengthscaleScale=1.0, periodScale=1.0, dtype=torch.double):
        self._scale = torch.tensor(scale, dtype=dtype)
        self._lengthscaleScale = lengthscaleScale
        self._periodScale = periodScale

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthscale, period = self._getAllParams()
        lengthscale = lengthscale/self._lengthscaleScale
        period = period/self._periodScale
        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            sDistance = X1-X2.transpose(1, 2)
        else:
            sDistance = X1.reshape(-1,1)-X2.reshape(1,-1)
        rr = math.pi*sDistance/period
        covMatrix = scale**2*torch.exp(-2*torch.sin(rr)**2/lengthscale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale, lengthscale, period = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        scale = self._scale
        lengthscale = self._params[0]
        period = self._params[1]

        return scale, lengthscale, period

    def getScaledParams(self):
        scaledParams = torch.tensor([self._params[0]/self._lengthscaleScale,
                                     self._params[1]/self._periodScale])
        return scaledParams

    def getNamedParams(self):
        scale, lengthscale, period = self._getAllParams()
        answer = {"scale": scale, "lengthscale": lengthscale, "period": period}
        return answer

