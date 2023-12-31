import sys
import os
import math
from scipy.io import loadmat
import numpy as np
import jax
import jax.numpy as jnp
import svGPFA.stats.kernels

jax.config.update("jax_enable_x64", True)

def test_exponentialQuadraticKernel_buildKernelMatrixX1():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/rbfKernel.mat")

    mat = loadmat(dataFilename)
    Z = mat['X1'].astype("float64").transpose((2,0,1))
    leasK = mat['G'].astype("float64").transpose((2,0,1))
    lengthscale = float(mat['lengthscale'][0,0])
    params = jnp.array([lengthscale])

    kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel()

    K = kernel.buildKernelMatrixX1(X1=Z, params=params)
    error = math.sqrt(((K-leasK)**2).flatten().mean())
    assert(error<tol)

    kernel_buildKernelMatrixX1_jitted = jax.jit(kernel.buildKernelMatrixX1)
    K = kernel_buildKernelMatrixX1_jitted(X1=Z, params=params)
    error = math.sqrt(((K-leasK)**2).flatten().mean())
    assert(error<tol)


def test_exponentialQuadraticKernel_buildKernelMatrixX1X2():
    tol = 1e-6
    k = 2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    assert(mat["kernelNames"][0, k][0] == "rbfKernel")

    Z = mat['Z'][k, 0].astype("float64").transpose((2,0,1))
    tt = mat['tt'].astype("float64").transpose((2,0,1))
    leasKtz = mat['Ktz'][k, 0].astype("float64").transpose((2,0,1))
    lengthscale = float(mat['hprs'][k][0][0,0])
    params = jnp.array([lengthscale])

    kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel()

    Ktz = kernel.buildKernelMatrixX1X2(X1=tt, X2=Z, params=params)
    error = math.sqrt(((Ktz-leasKtz)**2).flatten().mean())
    assert(error<tol)

    Ktz = kernel.buildKernelMatrixX1X2_jitted(X1=tt, X2=Z, params=params)
    error = math.sqrt(((Ktz-leasKtz)**2).flatten().mean())
    assert(error<tol)

def test_exponentialQuadraticKernelDiag():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Kdiag_rbfKernel.mat")

    mat = loadmat(dataFilename)
    t = mat['X1'].astype("float64").transpose(2,0,1)
    leasKDiag = mat['Gdiag'].astype("float64").transpose(2,0,1)
    lengthscale = float(mat['lengthscale'][0,0])
    scale = float(mat['variance'][0,0])
    params = jnp.array([lengthscale])

    kernel = svGPFA.stats.kernels.ExponentialQuadraticKernel(scale=scale)

    KDiag = kernel.buildKernelMatrixDiag(X=t, params=params)
    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())
    assert(error<tol)

    KDiag = kernel.buildKernelMatrixDiag_jitted(X=t, params=params)
    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())
    assert(error<tol)

def test_periodicKernel_buildKernelMatrixX1():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/PeriodicKernel.mat")

    mat = loadmat(dataFilename)
    Z = mat['X1'].astype("float64").transpose(2,0,1)
    leasK = mat['G'].astype("float64").transpose(2,0,1)
    lengthscale = float(mat['lengthscale'][0,0])
    period = float(mat['period'][0,0])
    params = jnp.array([lengthscale, period])

    kernel = svGPFA.stats.kernels.PeriodicKernel()

    K = kernel.buildKernelMatrixX1(X1=Z, params=params)
    error = math.sqrt(((K-leasK)**2).flatten().mean())
    assert(error<tol)

    K = kernel.buildKernelMatrixX1_jitted(X1=Z, params=params)
    error = math.sqrt(((K-leasK)**2).flatten().mean())
    assert(error<tol)

def test_periodicKernel_buildKernelMatrixX1X2():
    tol = 1e-6
    k = 0
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    assert(mat["kernelNames"][0, k][0] == "PeriodicKernel")

    Z = mat['Z'][k, 0].astype("float64").transpose((2,0,1))
    tt = mat['tt'].astype("float64").transpose((2,0,1))
    leasKtz = mat['Ktz'][k, 0].astype("float64").transpose((2,0,1))
    lengthscale = float(mat['hprs'][k][0][0,0])
    period = float(mat['hprs'][k][0][1,0])
    params = jnp.array([lengthscale, period])

    kernel = svGPFA.stats.kernels.PeriodicKernel()

    Ktz = kernel.buildKernelMatrixX1X2(X1=tt, X2=Z, params=params)
    error = math.sqrt(((Ktz-leasKtz)**2).flatten().mean())
    assert(error<tol)

    Ktz = kernel.buildKernelMatrixX1X2_jitted(X1=tt, X2=Z, params=params)
    error = math.sqrt(((Ktz-leasKtz)**2).flatten().mean())
    assert(error<tol)

def test_periodicKernelDiag():
    tol = 1e-6
    dataFilename = os.path.join(os.path.dirname(__file__), "data/Kdiag_PeriodicKernel.mat")

    mat = loadmat(dataFilename)
    t = mat['X1'].astype("float64").transpose(2,0,1)
    leasKDiag = mat['Gdiag'].astype("float64").transpose(2,0,1)
    lengthscale = float(mat['lengthscale'][0,0])
    period = float(mat['period'][0,0])
    scale = float(mat['variance'][0,0])
    params = jnp.array([lengthscale, period])

    kernel = svGPFA.stats.kernels.PeriodicKernel(scale=scale)

    KDiag = kernel.buildKernelMatrixDiag(X=t, params=params)
    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())
    assert(error<tol)

    KDiag = kernel.buildKernelMatrixDiag_jitted(X=t, params=params)
    error = math.sqrt(((KDiag-leasKDiag)**2).flatten().mean())
    assert(error<tol)

if __name__=="__main__":
    test_exponentialQuadraticKernel_buildKernelMatrixX1()
    test_exponentialQuadraticKernel_buildKernelMatrixX1X2()
    test_exponentialQuadraticKernelDiag()
    test_periodicKernel_buildKernelMatrixX1()
    test_periodicKernel_buildKernelMatrixX1X2()
    test_periodicKernelDiag()

