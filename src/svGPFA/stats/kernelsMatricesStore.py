
import jax
import jax.numpy as jnp


class IndPointsLocsKMS_Chol:

    def init(kernels: list):
        IndPointsLocsKMS_Chol.kernels = kernels

    @jax.jit
    def buildKernelsMatrices(kernels_params: list, ind_points_locs: jax.Array,
                             reg_param: float) -> tuple:
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        Kzz = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                               n_ind_points), dtype=jnp.double)
        Kzz_cho = jnp.empty(shape=(n_latents, n_trials, n_ind_points,
                                   n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            Kzz_k = (IndPointsLocsKMS_Chol.kernels[k].buildKernelMatrixX1(
                X1=ind_points_locs[k, :, :, :], params=kernels_params[k]) +
                reg_param * jnp.eye(N=n_ind_points, dtype=jnp.double))
            Kzz = Kzz.at[k, :, :, :].set(Kzz_k)
        Kzz_cho = IndPointsLocsKMS_Chol._cho3D(Kzz)
        return Kzz, Kzz_cho

    @jax.jit
    def _cho3D(Kzz: jax.Array):
        Kzz_cho = jnp.linalg.cholesky(Kzz)  # O(n^3)
        return Kzz_cho

    @jax.jit
    def solve(Kzz_cho: jax.Array, input: jax.Array) -> jax.Array:
        solve = jax.scipy.linalg.cho_solve((Kzz_cho, True), input)
        return solve


class IndPointsLocsAndQuadTimesKMS:

    def init(kernels: list, t: jax.Array):
        # t \in n_trials x n_spikes_all_neurons_per_trial
        IndPointsLocsAndQuadTimesKMS.kernels = kernels
        IndPointsLocsAndQuadTimesKMS.t = t

    @jax.jit
    def buildKernelsMatrices(kernels_params: list, ind_points_locs: jax.Array) -> jax.Array:
        # ind_points_locs \in nLatents x n_trials x n_ind_points x 1
        # return \in nLatents x n_trials x nQuadPoints x n_ind_points
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
            calculateKtzVMapped = jax.vmap(calculateKtz)
            Ktz_k = calculateKtzVMapped(IndPointsLocsAndQuadTimesKMS.t,
                                        ind_points_locs[k, :, :, :])
            Ktz = Ktz.at[k, :, :, :].set(Ktz_k)
        return Ktz


class IndPointsLocsAndSpikesTimesKMS:

    def init(kernels: list, t: jax.Array):
        # t \in n_trials x n_spikes
        IndPointsLocsAndSpikesTimesKMS.kernels = kernels
        IndPointsLocsAndSpikesTimesKMS.t = t

    @jax.jit
    def buildKernelsMatrices(kernels_params: list, ind_points_locs: jax.Array) -> jax.Array:
        # ind_points_locs \in nLatents x n_trials x n_ind_points x 1
        # return \in nLatents x n_trials x n_spikes x n_ind_points
        n_latents = ind_points_locs.shape[0]
        n_trials = ind_points_locs.shape[1]
        n_ind_points = ind_points_locs.shape[2]
        n_spikes = IndPointsLocsAndSpikesTimesKMS.t.shape[1]

        Ktz = jnp.empty(shape=(n_latents, n_trials, n_spikes,
                               n_ind_points), dtype=jnp.double)
        for k in range(n_latents):
            def calculateKtz(spikes_points, ind_points_locs):
                # n_spikes, n_ind_points -> [n_spikes, n_ind_points]
                Ktz = IndPointsLocsAndSpikesTimesKMS.kernels[k].buildKernelMatrixX1X2(
                    X1=spikes_points, X2=ind_points_locs,
                    params=kernels_params[k],
                )
                return Ktz
            # [n_trials, n_spikes], [n_trials, n_ind_points, 1] ->
            # [n_trials, n_spikes, n_ind_points]
            calculateKtzVMapped = jax.vmap(calculateKtz)
            Ktz_k = calculateKtzVMapped(IndPointsLocsAndSpikesTimesKMS.t,
                                        ind_points_locs[k, :, :, :])
            Ktz = Ktz.at[k, :, :, :].set(Ktz_k)
        return Ktz
