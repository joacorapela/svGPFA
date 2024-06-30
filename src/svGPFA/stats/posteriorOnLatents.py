
import jax
import jax.lax
import jax.numpy as jnp


class PosteriorOnLatents:

    @jax.jit
    def computeMeans(vMean, Kzz_cho, Ktz):
        # vMean \in n_latents x n_trials x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x (n_quad | n_spikes) x n_ind_points
        # return n_latents x n_trials x (n_quad | n_spikes)

        # ([n_ind_points, n_ind_points], [n_ind_points]) ->
        # [n_ind_points]
        def computeA(Kzz_cho, vMean):
            a = jax.scipy.linalg.cho_solve((Kzz_cho, True), vMean)
            return a

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points]) -> [n_trials, n_ind_points]
        computeA_vmTrials = jax.vmap(computeA, in_axes=(0, 0), out_axes=0)
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        #  [n_latents, n_trials, n_ind_points]) ->
        # [n_latents, n_trials, n_ind_points]
        computeA_vmLatents = jax.vmap(computeA_vmTrials, in_axes=(0, 0),
                                      out_axes=0)

        # A \in [n_latents x n_trials x n_ind_points]
        A = computeA_vmLatents(Kzz_cho, vMean)

        # ([n_quad, n_ind_points], [n_ind_points]) -> [n_quad]
        def computeMean(Ktz, A):
            answer = jnp.dot(Ktz, A)
            return answer

        # ([n_trials, n_quad, n_ind_points], [n_trials, n_ind_points]) ->
        # [n_trials, n_quad]
        computeMeans_vmTrials = jax.vmap(computeMean, in_axes=(0, 0))
        # ([n_latents,n_trials,  n_quad, n_ind_points],
        #  [n_latents, n_trials, n_ind_points]) ->
        # [n_latents, n_trials, n_quad]
        computeMeans_vmLatents = jax.vmap(computeMeans_vmTrials,
                                          in_axes=(0, 0))
        # qKMu\in [n_latents, n_trials, n_quad]
        qKMu = computeMeans_vmLatents(Ktz, A)

        return qKMu

    @jax.jit
    def computeVars(vCov, Kzz, Kzz_cho, Ktz, KttDiag=1.0):
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_cho \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_latents x n_trials x n_quad

        def computeVars(vCov, Kzz, Kzz_cho, Ktz, KttDiag):
            # vCov \in n_ind_points x n_ind_points
            # Kzz \in n_ind_points x n_ind_points
            # Kzz_cho \in n_ind_points x n_ind_points
            # Ktz \in n_quad x n_ind_points
            # answer n_quad

            # B \in n_ind_points x n_quad
            B = jax.scipy.linalg.cho_solve((Kzz_cho, True), Ktz.T)
            # mm1f \in n_ind_points x n_quad
            mm1f = jnp.matmul(vCov-Kzz, B)
            # aux1 \in n_ind_points x n_quad
            aux1 = B * mm1f
            # aux2 \in n_quad
            aux2 = jnp.sum(aux1, axis=0)
            # aux3 \in n_quad
            answer = KttDiag + aux2
            return answer

        # ([n_ind_points, n_ind_points], [n_ind_points, n_ind_points],
        # [n_ind_points, n_ind_points], [n_quad, n_ind_points], []) ->
        # [n_quad]
        # computeVars_vmQuadP = jax.vmap(computeVars,
        #                                in_axes=(None, None, None, 0, None))

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_quad, n_ind_points], []) ->
        # [n_trials, n_quad]
        computeVars_vmTrials = jax.vmap(computeVars,
                                        in_axes=(0, 0, 0, 0, None))
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_quad, n_ind_points], []) ->
        # [n_latents, n_trials, n_quad]
        computeVars_vmLatents = jax.vmap(computeVars_vmTrials,
                                         in_axes=(0, 0, 0, 0, None))
        qKVar = computeVars_vmLatents(vCov, Kzz, Kzz_cho, Ktz, KttDiag)
        return qKVar
