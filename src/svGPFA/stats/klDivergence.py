
import jax
import jax.numpy as jnp

class KLDivergence:

    def evalSumAcrossLatentsAndTrials(self, variational_mean, variational_cov,
                                      prior_cov, prior_cov_inv):
        klDiv = 0
        nLatents = len(variational_cov)
        for k in range(nLatents):
            klDivK = self._evalSumAcrossTrials(
                prior_cov=prior_cov[k],
                prior_cov_inv=prior_cov_inv[k],
                variational_mean=variational_mean[k],
                variational_cov=variational_cov[k],
                latent_index=k)
            klDiv += klDivK
        return klDiv

    def _evalSumAcrossTrials(self, prior_cov, prior_cov_inv,
                             variational_mean, variational_cov,
                             latent_index):
        # ESS \in n_trials x nInd x nInd
        ESS = variational_cov + jnp.matmul(variational_mean,
                                           variational_mean.transpose(0,2,1))
        n_trials = variational_mean.shape[0]
        answer = 0
        for trial_index in range(n_trials):
            _, prior_cov_logdet = jnp.linalg.slogdet(prior_cov[trial_index,:,:]) # O(n^3)
            _, variatioal_cov_logdet = jnp.linalg.slogdet(variational_cov[trial_index,:,:]) # O(n^3)
#             solve_term = self._indPointsLocsKMS.solveForLatentAndTrial(
#                 Kzz_inv=prior_cov_inv[trial_index,:,:],
#                 input=ESS[trial_index,:,:])
            solve_term = jax.scipy.linalg.cho_solve(
                (prior_cov_inv[trial_index,:,:], True), ESS[trial_index,:,:]
            )
            trace_term = jnp.trace(solve_term)
            trialKL = .5 * (trace_term + prior_cov_logdet -
                            variatioal_cov_logdet - ESS.shape[1])
            answer += trialKL
        return answer
