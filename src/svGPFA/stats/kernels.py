
from abc import ABC, abstractmethod
import math
import jax
import jax.numpy as jnp


class Kernel(ABC):

    @abstractmethod
    def buildKernelMatrixX1(self, X1):
        pass

    @abstractmethod
    def buildKernelMatrixX1X2(self, X1, X2):
        pass

    @abstractmethod
    def buildKernelMatrixDiag(self, X):
        pass


class ExponentialQuadraticKernel(Kernel):

    def buildKernelMatrixX1(self, X1, params):
        scale = params["scale"]
        lengthscale = params["lengthscale"]
        lengthscaleScale = params["lengthscaleScale"]
        lengthscale = lengthscale/lengthscaleScale

        X2 = X1
        if X1.ndim==3:
            distance = (X1-X2.transpose((0, 2, 1)))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*jnp.exp(-.5*distance/lengthscale**2)
        return covMatrix

    def buildKernelMatrixX1X2(self, X1, X2, params):
        scale = params["scale"]
        lengthscale = params["lengthscale"]
        lengthscaleScale = params["lengthscaleScale"]
        lengthscale = lengthscale/lengthscaleScale

        if X1.ndim==3:
            distance = (X1-X2.transpose((0, 2, 1)))**2
        else:
            distance = (X1.reshape(-1,1)-X2.reshape(1,-1))**2
        covMatrix = scale**2*jnp.exp(-.5*distance/lengthscale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X, params):
        scale = params["scale"]
        covMatrixDiag = scale**2*jnp.ones(X.shape, dtype=X.dtype)
        return covMatrixDiag


class PeriodicKernel(Kernel):

    def __init__(self):
        self.buildKernelMatrixX1_jitted = jax.jit(self.buildKernelMatrixX1)
        self.buildKernelMatrixX1X2_jitted = jax.jit(self.buildKernelMatrixX1X2)
        self.buildKernelMatrixDiag_jitted = jax.jit(self.buildKernelMatrixDiag)

    def buildKernelMatrixX1(self, X1, params):
        scale = params["scale"]
        lengthscale = params["lengthscale"]
        lengthscaleScale = params["lengthscaleScale"]
        period = params["period"]
        periodScale = params["periodScale"]
        lengthscale = lengthscale / lengthscaleScale
        period = period / periodScale
        X2 = X1
        if X1.ndim==3:
            sDistance = X1 - X2.transpose(0, 2, 1)
        else:
            sDistance = X1.reshape(-1,1) - X2.reshape(1,-1)
        rr = math.pi * sDistance / period
        covMatrix = scale**2 * jnp.exp(-2 * jnp.sin(rr)**2 / lengthscale**2)
        return covMatrix

    def buildKernelMatrixX1X2(self, X1, X2, params):
        scale = params["scale"]
        lengthscale = params["lengthscale"]
        lengthscaleScale = params["lengthscaleScale"]
        period = params["period"]
        periodScale = params["periodScale"]
        lengthscale = lengthscale / lengthscaleScale
        period = period / periodScale
        if X1.ndim==3:
            sDistance = X1 - X2.transpose(0, 2, 1)
        else:
            sDistance = X1.reshape(-1,1) - X2.reshape(1,-1)
        rr = math.pi * sDistance / period
        covMatrix = scale**2 * jnp.exp(-2 * jnp.sin(rr)**2 / lengthscale**2)
        return covMatrix

    def buildKernelMatrixDiag(self, X, params):
        scale = params["scale"]
        covMatrixDiag = scale**2 * jnp.ones(X.shape, dtype=X.dtype)
        return covMatrixDiag
