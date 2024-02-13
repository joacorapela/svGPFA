
import functools
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


    @functools.partial(jax.jit, static_argnums=0)
    def buildKernelsMatrices(self, kernels_params, ind_points_locs, reg_param):
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        Kzz = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                               n_ind_points), dtype=jnp.double)
        Kzz_inv = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                                   n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            Kzz = Kzz.at[k, :, :, :].set(self._kernels[k].buildKernelMatrixX1(
                X1=ind_points_locs[k, :, :, :], params=kernels_params[k]) +
                reg_param * jnp.eye(N=n_ind_points, dtype=jnp.double))
        Kzz_inv = self._invertKzz3D(Kzz)
        return Kzz, Kzz_inv


class IndPointsLocsKMS_Chol(IndPointsLocsKMS):

    @functools.partial(jax.jit, static_argnums=0)
    def _invertKzz3D(self, Kzz):
        Kzz_chol = jnp.linalg.cholesky(Kzz) # O(n^3)
        return Kzz_chol

    @functools.partial(jax.jit, static_argnums=0)
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

    def __init__(self, kernels, times):
        super().__init__(kernels=kernels)
        self._t = times

    def getKtz(self):
        return self._Ktz

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

class IndPointsLocsAndQuadTimesKMS(IndPointsLocsAndTimesKMS):

    # self._t \in n_trials x n_quad_points

    @functools.partial(jax.jit, static_argnums=0)
    def buildKernelsMatrices(self, kernels_params, ind_points_locs):
        # ind_points_locs \in nLatents x nTrials x nIndPoints x 1
        # return \in nLatexts x nTrials x nQuadPoints x nIndPoints
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        n_quad_points = self._t.shape[1]

        Ktz = jnp.empty(shape=(n_latents, n_trials, n_quad_points,
                               n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            def calculateKtz(quad_points, ind_points_locs):
                Ktz = self._kernels[k].buildKernelMatrixX1X2(
                    X1=quad_points, X2=ind_points_locs,
                    params=kernels_params[k],
                )
                return Ktz
            calculateKtzVMapped = jax.vmap(calculateKtz, in_axes=(0, 0))
            Ktz_k = calculateKtzVMapped(self._t, ind_points_locs[k, :, :, :])
            Ktz = Ktz.at[k, :, :, :].set(Ktz_k)
        return Ktz


class IndPointsLocsAndSpikesTimesKMS(IndPointsLocsAndTimesKMS):

    # self._t[r] \in n_spikes[r] // n_spikes from all neurons concatenated

    # don't jit due to the problem of unrollling of the trials loop
    def buildKernelsMatrices(self, kernels_params, ind_points_locs):
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]

        Ktz = [None for r in len(n_trials)]
        for r in range(n_trials):
            n_spikes_r = len(self._t[r])
            Ktz[r] = jnp.empty(shape=(n_latents, n_spikes_r, n_ind_points),
                               dtype=jnp.double)
            for k in range(n_latents):
                Ktz_kr = self._kernels[k].buildKernelMatrixX1X2(
                    X1=self._t[r], X2=ind_points_locs[k, r, :],
                )
                Ktz[r] = Ktz[r].at[k, :, :].set(Ktz_kr)
        return Ktz
