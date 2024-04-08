
import jax
import jax.numpy as jnp


class IndPointsLocsKMS_Chol:

    def init(kernels):
        IndPointsLocsKMS_Chol.kernels = kernels

    @jax.jit
    def buildKernelsMatrices(kernels_params, ind_points_locs, reg_param):
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        Kzz = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                               n_ind_points), dtype=jnp.double)
        Kzz_inv = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                                   n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            Kzz_k = (IndPointsLocsKMS_Chol.kernels[k].buildKernelMatrixX1(
                X1=ind_points_locs[k, :, :, :], params=kernels_params[k]) +
                reg_param * jnp.eye(N=n_ind_points, dtype=jnp.double))
            Kzz = Kzz.at[k, :, :, :].set(Kzz_k)
        Kzz_inv = IndPointsLocsKMS_Chol._invertKzz3D(Kzz)
        return Kzz, Kzz_inv

    @jax.jit
    def _invertKzz3D(Kzz):
        Kzz_chol = jnp.linalg.cholesky(Kzz)  # O(n^3)
        return Kzz_chol

    @jax.jit
    def solve(Kzz_inv, input):
        solve = jax.scipy.linalg.cho_solve((Kzz_inv, True), input)
        return solve


class IndPointsLocsAndQuadTimesKMS:

    def init(kernels, t):
        IndPointsLocsAndQuadTimesKMS.kernels = kernels
        IndPointsLocsAndQuadTimesKMS.t = t

    @jax.jit
    def buildKernelsMatrices(kernels_params, ind_points_locs):
        # t \in n_trials x n_quad_points
        # ind_points_locs \in nLatents x n_trials x n_ind_points x 1
        # return \in nLatexts x n_trials x nQuadPoints x n_ind_points
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        n_quad_points = IndPointsLocsAndQuadTimesKMS.t.shape[1]

        Ktz = jnp.empty(shape=(n_latents, n_trials, n_quad_points,
                               n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            def calculateKtz(quad_points, ind_points_locs):
                # n_quad_points, n_ind_points -> [n_quad_points, n_ind_points]
                Ktz = IndPointsLocsAndQuadTimesKMS.kernels[k].buildKernelMatrixX1X2(
                    X1=quad_points, X2=ind_points_locs,
                    params=kernels_params[k],
                )
                return Ktz
            # [n_trials, n_quad_points], [n_trials, n_ind_points, 1] ->
            # [n_trials, n_quad_points, n_ind_points]
            calculateKtzVMapped = jax.vmap(calculateKtz, in_axes=(0, 0))
            Ktz_k = calculateKtzVMapped(IndPointsLocsAndQuadTimesKMS.t,
                                        ind_points_locs[k, :, :, :])
            Ktz = Ktz.at[k, :, :, :].set(Ktz_k)
        return Ktz


class IndPointsLocsAndSpikesTimesKMS:

    def init(kernels, t, t_all):
        IndPointsLocsAndSpikesTimesKMS.kernels = kernels
        IndPointsLocsAndSpikesTimesKMS.t = t
        IndPointsLocsAndSpikesTimesKMS.t_all = t_all

    # don't jit due to the problem of unrollling of the trials loop
    def buildKernelsMatrices(kernels_params, ind_points_locs):
        # t[r] \in n_spikes[r] // n_spikes from all neurons concatenated
        # ind_points_locs \in n_ind_points x 1
        # answer Ktz[r] \in n_latents x n_spikes_all_neurons[r] x n_ind_points
        n_latents = len(IndPointsLocsAndSpikesTimesKMS.kernels)
        n_trials = len(IndPointsLocsAndSpikesTimesKMS.t)
        n_ind_points = ind_points_locs.shape[0]

        Ktzc = jnp.empty(shape=(n_latents, len(IndPointsLocsAndSpikesTimesKMS.t_all), n_ind_points),
                         dtype=jnp.double)
        for k in range(n_latents):
            Ktzc_k = IndPointsLocsAndSpikesTimesKMS.kernels[k].buildKernelMatrixX1X2(
                X1=IndPointsLocsAndSpikesTimesKMS.t_all,
                X2=ind_points_locs,
                params=kernels_params[k],
            )
            Ktzc = Ktzc.at[k, :, :].set(Ktzc_k)

        Ktz = [None for r in range(n_trials)]
        base_index = 0
        for r in range(n_trials):
            Ktz[r] = Ktzc[:,
                          slice(base_index,
                                base_index+len(IndPointsLocsAndSpikesTimesKMS.t[r])),
                          :]
            base_index += len(IndPointsLocsAndSpikesTimesKMS.t[r])
        return Ktz
