
import jax
import jax.numpy as jnp


class ExponentialQuadraticKernel:

    def buildKernelMatrixX1(X1: jax.Array, params: list, scale: float = 1.0) -> jax.Array:
        return ExponentialQuadraticKernel.buildKernelMatrixX1X2(
            X1=X1, X2=X1, params=params, scale=scale)

    def buildKernelMatrixX1X2(X1: jax.Array, X2: jax.Array, params: list,
                              scale: float = 1.0):
        lengthscale = params[0]

        distance = (X1-jnp.swapaxes(X2, -1, -2))**2
        covMatrix = scale**2*jnp.exp(-.5*distance/lengthscale**2)
        return covMatrix


class ExponentialQuadraticKernelSigmaAsParam:

    def buildKernelMatrixX1(X1: jax.Array, params: list) -> jax.Array:
        return ExponentialQuadraticKernelSigmaAsParam.buildKernelMatrixX1X2(
            X1=X1, X2=X1, params=params)

    def buildKernelMatrixX1X2(X1: jax.Array, X2: jax.Array, params: list):
        sigma = params[0]**2
        lengthscale = params[1]

        distance = (X1-jnp.swapaxes(X2, -1, -2))**2
        covMatrix = sigma**2*jnp.exp(-.5*distance/lengthscale**2)
        return covMatrix

class ExponentialQuadraticLogLengthscaleKernel:
    def buildKernelMatrixX1(X1: jax.Array, params: list, scale: float = 1.0) -> jax.Array:
        log_params = [jax.log(param) for param in params]
        return ExponentialQuadraticKernel.buildKernelMatrixX1X2(
            X1=X1, X2=X1, params=log_params, scale=scale)

    def buildKernelMatrixX1X2(X1: jax.Array, X2: jax.Array, params: list,
                              scale: float = 1.0):
        log_params = [jax.log(param) for param in params]
        return ExponentialQuadraticKernel.buildKernelMatrixX1X2(
            X1=X1, X2=X1, params=log_params, scale=scale)


class PeriodicKernel:

    def buildKernelMatrixX1(X1: jax.Array, params: list, scale: float = 1.0) -> jax.Array:
        return PeriodicKernel.buildKernelMatrixX1X2(X1=X1, X2=X1,
                                                    params=params, scale=scale)

    def buildKernelMatrixX1X2(X1: jax.Array, X2: jax.Array, params: list,
                              scale: float = 1.0) -> jax.Array:
        lengthscale = params[0]
        period = params[1]
        distance = jnp.abs(X1-jnp.swapaxes(X2, -1, -2))
        rr = jnp.pi * distance / period
        covMatrix = scale**2 * jnp.exp(-2 * jnp.sin(rr)**2 / lengthscale**2)
        return covMatrix
