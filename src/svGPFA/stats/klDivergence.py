
import jax
import jax.numpy as jnp


class KLDivergence:

    def evalSumAcrossLatentsAndTrials(vMean, vCov, Kzz, Kzz_cho):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points

        def _computeKL(vMean, vCov, Kzz, Kzz_cho):
            # vMean   \in n_ind_points
            # vCov    \in n_ind_points x n_ind_points
            # Kzz     \in n_ind_points x n_ind_points
            # Kzz_cho \in n_ind_points x n_ind_points
            K = len(vMean)
            ESS = vCov + jnp.outer(vMean, vMean)
            _, Kzz_logdet = jnp.linalg.slogdet(Kzz)    # O(n^3)
            _, vCov_logdet = jnp.linalg.slogdet(vCov)  # O(n^3)
            solve_term = jax.scipy.linalg.cho_solve((Kzz_cho, True), ESS)
            trace_term = jnp.trace(solve_term)
            kl = .5 * (trace_term + Kzz_logdet - vCov_logdet - K)
            return kl

        _vmap_trials = jax.vmap(_computeKL)
        _vmap_latents  = jax.vmap(_vmap_trials)

        kl_latents_trials = _vmap_latents(vMean, vCov, Kzz, Kzz_cho)
        kl_sum = jnp.sum(kl_latents_trials)
        return kl_sum
