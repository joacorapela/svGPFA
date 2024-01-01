
import os
import math
from scipy.io import loadmat
import jax
import jax.numpy as jnp
import svGPFA.stats.kernels
import svGPFA.stats.kernelsMatricesStore

jax.config.update("jax_enable_x64", True)

def test_eval_IndPointsLocsKMS():
    tol = 1e-5
    # tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    leasKzz = [jax.device_put(mat['Kzz'][(i,0)].astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    # leasKzzi = [mat['Kzzi'][(i,0)].astype("float64").transpose(2,0,1) for i in range(nLatents)]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]
    epsilon = mat["epsilon"][0,0]

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0, k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsKMS_Chol(
        kernels=kernels)
    Kzz, Kzz_chol = indPointsLocsKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0,
        reg_param=epsilon)

    for k in range(len(Kzz)):
        error = math.sqrt(((Kzz[k]-leasKzz[k])**2).flatten().mean())
        assert(error<tol)

# It is difficult to test Kzz_inv, since Lea was using pinv and we are using the
# Cholesky decomposition
#
#     for k in range(len(Kzz_chol)):
#         Kzz_inv_k = jnp.linalg.inv(jnp.matmul(Kzz_chol[k],
#                                             jnp.transpose(Kzz_chol[k], axes=(0, 2, 1))))
#         error = math.sqrt(((Kzz_inv_k-leasKzzi[k])**2).flatten().mean())
#         breakpoint()
#     assert(error<tolKzzi)


def test_eval_IndPointsLocsAndQuadTimesKMS():
    tol = 1e-5
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    t = jax.device_put((mat['tt']).astype("float64").transpose(2, 0, 1))
    Z0 = [jax.device_put((mat['Z'][(i,0)]).astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    leasKtz = [jax.device_put((mat['Ktz'][(i,0)]).astype("float64").transpose(2,0,1)) for i in range(nLatents)]
    leasKttDiag = jax.device_put((mat['Ktt']).astype("float64").transpose(2, 0, 1))
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0, k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsAndQuadTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=t)
    estKtz, estKttDiag = indPointsLocsAndQuadTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    for k in range(nLatents):
        for r in range(nTrials):
            error = math.sqrt(((estKtz[k][r]-leasKtz[k][r,:,:])**2).flatten().mean())
            assert(error<tol)
            error = math.sqrt(((estKttDiag[k][r]-leasKttDiag[r,:,k])**2).flatten().mean())
            assert(error<tol)


def test_eval_IndPointsLocsAndSpikesTimesKMS():
    tol = 1e-5
    # tolKzzi = 6e-2
    dataFilename = os.path.join(os.path.dirname(__file__), "data/BuildKernelMatrices_fromSpikes.mat")

    mat = loadmat(dataFilename)
    nLatents = mat['Z'].shape[0]
    nTrials = mat['Z'][0,0].shape[2]
    Y =  [jax.device_put(mat['Y'][(r,0)].astype("float64")) for r in range(nTrials)]
    Z0 = [jax.device_put(mat['Z'][(i,0)].astype("float64")).transpose(2,0,1) for i in range(nLatents)]
    leasKtz = [[jax.device_put(mat['Ktz'][i,j].astype("float64")) for j in range(nTrials)] for i in range(nLatents)]
    leasKttDiag = [[jax.device_put(mat['Ktt'][i,j].astype("float64")) for j in range(nTrials)] for i in range(nLatents)]
    kernelNames = mat["kernelNames"]
    hprs = mat["hprs"]

    kernels = [[None] for k in range(nLatents)]
    kernels_params0 = [[None] for k in range(nLatents)]
    for k in range(nLatents):
        if kernelNames[0, k][0] == "PeriodicKernel":
            kernels[k] = svGPFA.stats.kernels.PeriodicKernel()
            lengthscale = float(hprs[k,0][0].item())
            period = float(hprs[k,0][1].item())
            kernels_params0[k] = jnp.array([lengthscale, period])
        elif kernelNames[0,k][0] == "rbfKernel":
            kernels[k] = svGPFA.stats.kernels.ExponentialQuadraticKernel()
            lengthscale = float(hprs[k,0][0].item())
            kernels_params0[k] = jnp.array([lengthscale])
        else:
            raise ValueError("Invalid kernel name: %s"%(kernelNames[k]))

    indPointsLocsAndSpikesTimesKMS = svGPFA.stats.kernelsMatricesStore.IndPointsLocsAndTimesKMS(
        kernels=kernels, times=Y)
    estKtz, estKttDiag = indPointsLocsAndSpikesTimesKMS.buildKernelsMatrices(
        kernels_params=kernels_params0, ind_points_locs=Z0)

    for k in range(nLatents):
        for r in range(nTrials):
            error = math.sqrt(((estKtz[k][r]-leasKtz[k][r])**2).flatten().mean())
            assert(error<tol)
            error = math.sqrt(((estKttDiag[k][r]-leasKttDiag[k][r])**2).flatten().mean())
            assert(error<tol)


if __name__=='__main__':
    test_eval_IndPointsLocsKMS()
    test_eval_IndPointsLocsAndQuadTimesKMS()
    test_eval_IndPointsLocsAndSpikesTimesKMS()
