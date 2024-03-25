
import functools
import jax
import jax.lax
import jax.numpy as jnp


class PosteriorOnLatentsQuad:

    @functools.partial(jax.jit, static_argnums=0)
    def computeMeansAndVars(variational_mean, variational_cov,
                            Kzz, Kzz_inv, Ktz, KttDiag=1.0):
        # variational_mean \in n_latents x n_trials x n_ind_points x 1
        # variational_cov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_latents x n_trials x n_quad

        # ([n_ind_points, n_ind_points], [n_ind_points, 1]) ->
        # [n_ind_points, 1]
        def computeA(Kzz_inv, variational_mean):
            a = jax.scipy.linalg.cho_solve((Kzz_inv, True), variational_mean)
            return a

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points, 1]) -> [n_trials, n_ind_points, 1]
        computeA_vmTrials = jax.vmap(computeA, in_axes=(0, 0), out_axes=0)
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        #  [n_latents, n_trials, n_ind_points, 1]) ->
        # [n_latents, n_trials, n_ind_points, 1]
        computeA_vmLatents = jax.vmap(computeA_vmTrials, in_axes=(0, 0),
                                      out_axes=0)

        # A \in [n_latents x n_trials x n_ind_points x 1]
        A = computeA_vmLatents(Kzz_inv, variational_mean)

        # ([n_quad, n_ind_points], [n_ind_points, 1]) -> [n_quad, 1]
        def computeMean(Ktz, A):
            answer = jnp.dot(Ktz, A)
            return answer
        # computeMeans_vmQuad = jax.vmap(computeMean, in_axes=(0, None))
        # computeMeans_vmTrials = jax.vmap(computeMeans_vmQuad, in_axes=(0, 0))

        # ([n_trials, n_quad, n_ind_points], [n_trials, n_ind_points]) ->
        # [n_trials, n_quad, 1]
        computeMeans_vmTrials = jax.vmap(computeMean, in_axes=(0, 0))
        # ([n_latents,n_trials,  n_quad, n_ind_points],
        #  [n_latents, n_trials, n_ind_points, 1]) ->
        # [n_latents, n_trials, n_quad, 1]
        computeMeans_vmLatents = jax.vmap(computeMeans_vmTrials,
                                          in_axes=(0, 0))
        # means\in [n_latents, n_trials, n_quad, 1]
        means = computeMeans_vmLatents(Ktz, A)

        def computeVars(variational_cov, Kzz, Kzz_inv, Ktz, KttDiag):
            # variational_cov \in n_ind_points x n_ind_points
            # Kzz \in n_ind_points x n_ind_points
            # Kzz_inv \in n_ind_points x n_ind_points
            # Ktz \in n_ind_points

            # B \in n_ind_points
            B = jax.scipy.linalg.cho_solve((Kzz_inv, True), Ktz)
            # diff \in n_ind_points x n_ind_points
            diff = variational_cov - Kzz
            # std \in \Re
            sigma2 = KttDiag + jnp.dot(B, jnp.matmul(diff, B))
            return sigma2

        # ([n_ind_points, n_ind_points], [n_ind_points, n_ind_points],
        # [n_ind_points, n_ind_points], [n_quad, n_ind_points], []) ->
        # [n_quad]
        computeVars_vmQuadP = jax.vmap(computeVars,
                                       in_axes=(None, None, None, 0, None))
        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_ind_points, n_ind_points],
        # [n_trials, n_quad, n_ind_points], []) ->
        # [n_trials, n_quad]
        computeVars_vmTrials = jax.vmap(computeVars_vmQuadP,
                                        in_axes=(0, 0, 0, 0, None))
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_ind_points, n_ind_points],
        # [n_latents, n_trials, n_quad, n_ind_points], []) ->
        # [n_latents, n_trials, n_quad]
        computeVars_vmLatents = jax.vmap(computeVars_vmTrials,
                                         in_axes=(0, 0, 0, 0, None))
        sigma2s = computeVars_vmLatents(variational_cov, Kzz, Kzz_inv, Ktz,
                                        KttDiag)
        return means, sigma2s


class PosteriorOnLatentsSpikes:

    @functools.partial(jax.jit, static_argnums=0)
    def computeTrialMeansAndVars(variational_mean, variational_cov,
                                 Kzz, Kzz_inv, Ktz, KttDiag):
        # variational_mean \in n_latents x n_ind_points x 1
        # variational_cov \in n_latents x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_spikes[r] x n_ind_points
        # KttDiag \in Real
        # return (n_spikes_r x n_latents, n_spikes_r x n_latents)

        def computeLatentMeansAndVars(variational_mean, variational_cov,
                                      Kzz, Kzz_inv, Ktz, KttDiag):
            Akr = jax.scipy.linalg.cho_solve((Kzz_inv[:, :], True),
                                             variational_mean[:, :])
            qKMu = jnp.squeeze(jnp.matmul(Ktz[:, :], Akr))

            # Bfk \in n_ind_points x n_spikes_r[r]
            Bfk = jax.scipy.linalg.cho_solve((Kzz_inv[:, :], True),
                                             Ktz[:, :].transpose((1, 0)))

            # mm1f \in n_ind_points x n_spikes_r[r]
            diff = variational_cov[:, :]-Kzz[:, :]
            mm1f = jnp.matmul(diff, Bfk)

            # qKVar[r] \in nTimes[r] x n_latents
            qKVar = KttDiag + jnp.sum(a=Bfk*mm1f, axis=0)

            return qKMu, qKVar

        computeLatentMeansAndVars_vmLatents = jax.vmap(
            computeLatentMeansAndVars, (0, 0, 0, 0, 0, None), (1, 1))
        qKMu, qKVar = computeLatentMeansAndVars_vmLatents(
            variational_mean, variational_cov, Kzz, Kzz_inv, Ktz, KttDiag)
        return qKMu, qKVar

    def computeMeansAndVars(variational_mean, variational_cov,
                            Kzz, Kzz_inv, Ktz, KttDiag=1.0):
        # variational_mean \in n_latents x n_trials x n_ind_points x 1
        # variational_cov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz[r] \in n_latents x n_spikes[r] x n_ind_points
        # KttDiag \in Real
        # return list[r] \in n_spikes_r x n_latents

        n_trials = variational_mean.shape[1]
        qKMu = [[None] for tr in range(n_trials)]
        qKVar = [[None] for tr in range(n_trials)]
        for r in range(n_trials):
            # qKMu[r] \in nTimes[r] x n_latents
            qKMu[r], qKVar[r] = \
                PosteriorOnLatentsSpikes.computeTrialMeansAndVars(
                    variational_mean=variational_mean[:, r, :, :],
                    variational_cov=variational_cov[:, r, :, :],
                    Kzz=Kzz[:, r, :, :], Kzz_inv=Kzz_inv[:, r, :, :],
                    Ktz=Ktz[r], KttDiag=KttDiag)
        return qKMu, qKVar
