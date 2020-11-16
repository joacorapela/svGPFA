
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

    def __init__(self, scale, dtype=torch.double, device=torch.device("cpu")):
        self._scale = torch.tensor(scale)

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthScale = self._getAllParams()

        if X2 is None:
            X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose(1, 2))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*torch.exp(-.5*distance/lengthScale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X):
        scale, lengthScale = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        scale = self._scale
        lengthScale = self._params[0]
        return scale, lengthScale

    def getNamedParams(self):
        scale, lengthScale = self._getAllParams()
        answer = {"scale": scale, "lengthScale": lengthScale}
        return answer

class PeriodicKernel(Kernel):
    def __init__(self, scale, dtype=torch.double, device=torch.device("cpu")):
        self._scale = torch.tensor(scale)

    def buildKernelMatrix(self, X1, X2=None):
        scale, lengthScale, period = self._getAllParams()
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
        scale, lengthScale, period = self._getAllParams()
        covMatrixDiag = scale**2*torch.ones(X.shape, dtype=X.dtype, device=X.device)
        return covMatrixDiag

    def _getAllParams(self):
        scale = self._scale
        lengthScale = self._params[0]
        period = self._params[1]

        return scale, lengthScale, period

    def getNamedParams(self):
        scale, lengthScale, period = self._getAllParams()
        answer = {"scale": scale, "lengthScale": lengthScale, "period": period}
        return answer

class ParamsScaledKernel(Kernel):
    def __init__(self, baseKernel, paramsScales, dtype=torch.double, device=torch.device("cpu")):
        self._baseKernel = baseKernel
        self._paramsScales = paramsScales

    def setParams(self, params):
        unscaledParams = params/torch.tensor(self._paramsScales)
        self._baseKernel.setParams(params=unscaledParams)

    def getParams(self):
        unscaledParams = self._baseKernel.getParams()
        scaledParams = unscaledParams*torch.tensor(self._paramsScales)
        return scaledParams

    def buildKernelMatrix(self, X1, X2=None):
        return self._baseKernel.buildKernelMatrix(X1=X1, X2=X2)

    def buildKernelMatrixDiag(self, X):
        return self._baseKernel.buildKernelMatrixDiag(X=X)

    def _getAllParams(self):
        return self._baseKernel._getAllParams()

    def getNamedParams(self):
        return self._baseKernel._getNamedParams()

