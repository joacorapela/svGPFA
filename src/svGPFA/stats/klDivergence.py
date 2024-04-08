
import jax
import jax.numpy as jnp


class KLDivergence:

    def evalSumAcrossLatentsAndTrials(vMean, vCov, Kzz, Kzz_inv):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points

        def _computeKL(vMean, vCov, Kzz, Kzz_inv):
            # vMean   \in n_ind_points
            # vCov    \in n_ind_points x n_ind_points
            # Kzz     \in n_ind_points x n_ind_points
            # Kzz_inv \in n_ind_points x n_ind_points
            K = len(vMean)
            ESS = vCov + jnp.matmul(vMean, vMean.T)
            _, Kzz_logdet = jnp.linalg.slogdet(Kzz)    # O(n^3)
            _, vCov_logdet = jnp.linalg.slogdet(vCov)  # O(n^3)
            solve_term = jax.scipy.linalg.cho_solve((Kzz_inv, True), ESS)
            trace_term = jnp.trace(solve_term)
            kl = .5 * (trace_term + Kzz_logdet - vCov_logdet - K)
            return kl

        _computeKL_latents = jax.vmap(_computeKL)
        _computeKL_trials  = jax.vmap(_computeKL_latents)

        kl_trails_latents = _computeKL_trials(vMean, vCov, Kzz, Kzz_inv)
        kl_sum = jnp.sum(kl_trails_latents)
        return kl_sum
