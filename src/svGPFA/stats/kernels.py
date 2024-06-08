
import jax
import jax.numpy as jnp


class ExponentialQuadraticKernel:

    @jax.jit
    def buildKernelMatrixX1(X1, params, scale=1.0):
        return ExponentialQuadraticKernel.buildKernelMatrixX1X2(
            X1=X1, X2=X1, params=params, scale=scale)

    @jax.jit
    def buildKernelMatrixX1X2(X1, X2, params, scale=1.0):
        lengthscale = params[0]

        distance = (X1-jnp.swapaxes(X2, -1, -2))**2
        covMatrix = scale**2*jnp.exp(-.5*distance/lengthscale**2)
        return covMatrix


class PeriodicKernel:

    @jax.jit
    def buildKernelMatrixX1(X1, params, scale=1.0):
        return PeriodicKernel.buildKernelMatrixX1X2(X1=X1, X2=X1,
                                                    params=params, scale=scale)

    @jax.jit
    def buildKernelMatrixX1X2(X1, X2, params, scale=1.0):
        lengthscale = params[0]
        period = params[1]
        distance = (X1-jnp.swapaxes(X2, -1, -2))**2
        rr = jnp.pi * distance / period
        covMatrix = scale**2 * jnp.exp(-2 * jnp.sin(rr)**2 / lengthscale**2)
        return covMatrix
