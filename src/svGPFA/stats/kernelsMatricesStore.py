
from functools import partial
from jax import jit
import jax
import jax.numpy as jnp
import abc
import svGPFA.utils.miscUtils


class KernelsMatricesStore(abc.ABC):

#     @abc.abstractmethod
#     def buildKernelsMatrices(self):
#         pass

    def __init__(self, kernels):
        self._kernels = kernels

#     def setKernels(self, kernels):
#         self._kernels = kernels

    def setIndPointsLocs(self, ind_points_locs):
        self._ind_points_locs = ind_points_locs

    def getIndPointsLocs(self):
        return self._ind_points_locs

    def getKernels(self):
        return self._kernels


class IndPointsLocsKMS(KernelsMatricesStore):

    # @abc.abstractmethod
    def _invertKzz3D(self, Kzz):
        pass

#     @abc.abstractmethod
#     def solveForLatent(self, Kzz_inv, input, latent_index):
        pass

#     @abc.abstractmethod
#     def solveForLatentAndTrial(self, Kzz_inv, input, latent_index, trial_index):
        pass

    def buildKernelsMatrices(self, kernels_params, ind_points_locs, reg_param):
        n_latents = len(kernels_params)
        Kzz = [[None] for k in range(n_latents)]
        Kzz_inv = [[None] for k in range(n_latents)]

        for k in range(n_latents):
            Kzz[k] = (self._kernels[k].buildKernelMatrixX1(X1=ind_points_locs[k],
                                                     params=kernels_params[k]) +
                      reg_param * jnp.eye(N=ind_points_locs[k].shape[1],
                                          dtype=ind_points_locs[k].dtype))
            Kzz_inv[k] = self._invertKzz3D(Kzz[k]) # O(n^3)
        return Kzz, Kzz_inv


class IndPointsLocsKMS_Chol(IndPointsLocsKMS):

    def _invertKzz3D(self, Kzz):
        Kzz_chol = jnp.linalg.cholesky(Kzz) # O(n^3)
        return Kzz_chol

    def solve(self, Kzz_inv, input):
        solve = jax.scipy.linalg.cho_solve((Kzz_inv, True), input)
        return solve


class IndPointsLocsKMS_PInv(IndPointsLocsKMS):

    def _invertKzz3D(self, Kzz):
        Kzz_inv = svGPFA.utils.miscUtils.pinv3D(Kzz)  # O(n^3)
        return Kzz_inv

    def solveForLatent(self, input, latent_index):
        solve = torch.matmul(self._Kzz_inv[latent_index], input)
        return solve

    def solveForLatentAndTrial(self, input, latent_index, trial_index):
        solve = torch.matmul(self._Kzz_inv[latent_index][trial_index, :, :],
                             input)
        return solve


class IndPointsLocsAndTimesKMS(KernelsMatricesStore):

    def setTimes(self, times):
        # times[r] \in nTimes x 1
        self._t = times

    def getKtz(self):
        return self._Ktz

    def getKtt(self):
        return self._Ktt

    def getKttDiag(self):
        return self._KttDiag

#     def buildKernelsMatrices(self):
#         n_latents = len(self._ind_points_locs)
#         n_trials = self._ind_points_locs[0].shape[0]
#         self._Ktz = [[[None] for tr in range(n_trials)]
#                      for k in range(n_latents)]
#         self._KttDiag = [[[None] for tr in range(n_trials)] for k in
#                          range(n_latents)]
# 
#         for k in range(n_latents):
#             for tr in range(n_trials):
#                 self._Ktz[k][tr] = self._kernels[k].buildKernelMatrixX1X2(
#                     X1=self._t[tr], X2=self._ind_points_locs[k][tr, :, :])
#                 self._KttDiag[k][tr] = self._kernels[k].buildKernelMatrixDiag(
#                     X=self._t[tr])

    def buildKernelsMatrices(self, kernels_params, ind_points_locs):
        n_latents = len(ind_points_locs)
        n_trials = ind_points_locs[0].shape[0]
        Ktz = [[[None] for tr in range(n_trials)]
                     for k in range(n_latents)]
        KttDiag = [[[None] for tr in range(n_trials)] for k in
                         range(n_latents)]

        for k in range(n_latents):
            for tr in range(n_trials):
                Ktz[k][tr] = self._kernels[k].buildKernelMatrixX1X2(
                    X1=self._t[tr], X2=ind_points_locs[k][tr, :, :],
                    params=kernels_params[k])
                KttDiag[k][tr] = self._kernels[k].buildKernelMatrixDiag(
                    X=self._t[tr], params=kernels_params[k])

        return Ktz, KttDiag
