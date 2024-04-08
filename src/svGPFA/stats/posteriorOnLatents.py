
import jax
import jax.lax
import jax.numpy as jnp


class PosteriorOnLatentsQuad:

    @jax.jit
    def computeMeansAndVars(vMean, vCov, Kzz, Kzz_inv, Ktz, KttDiag=1.0):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_trials x n_quad x n_ind_points
        # KttDiag \in Reals
        # return n_latents x n_trials x n_quad

        # ([n_ind_points, n_ind_points], [n_ind_points]) ->
        # [n_ind_points]
        def computeA(Kzz_inv, vMean):
            a = jax.scipy.linalg.cho_solve((Kzz_inv, True), vMean)
            return a

        # ([n_trials, n_ind_points, n_ind_points],
        #  [n_trials, n_ind_points, 1]) -> [n_trials, n_ind_points]
        computeA_vmTrials = jax.vmap(computeA, in_axes=(0, 0), out_axes=0)
        # ([n_latents, n_trials, n_ind_points, n_ind_points],
        #  [n_latents, n_trials, n_ind_points]) ->
        # [n_latents, n_trials, n_ind_points]
        computeA_vmLatents = jax.vmap(computeA_vmTrials, in_axes=(0, 0),
                                      out_axes=0)

        # A \in [n_latents x n_trials x n_ind_points]
        A = computeA_vmLatents(Kzz_inv, vMean)

        # ([n_quad, n_ind_points], [n_ind_points]) -> [n_quad]
        def computeMean(Ktz, A):
            answer = jnp.dot(Ktz, A)
            return answer
        # computeMeans_vmQuad = jax.vmap(computeMean, in_axes=(0, None))
        # computeMeans_vmTrials = jax.vmap(computeMeans_vmQuad, in_axes=(0, 0))

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

        def computeVars(vCov, Kzz, Kzz_inv, Ktz, KttDiag):
            # vCov \in n_ind_points x n_ind_points
            # Kzz \in n_ind_points x n_ind_points
            # Kzz_inv \in n_ind_points x n_ind_points
            # Ktz \in n_ind_points

            # B \in n_ind_points
            B = jax.scipy.linalg.cho_solve((Kzz_inv, True), Ktz)
            # diff \in n_ind_points x n_ind_points
            diff = vCov - Kzz
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
        qKVar = computeVars_vmLatents(vCov, Kzz, Kzz_inv, Ktz, KttDiag)
        return qKMu, qKVar


class PosteriorOnLatentsSpikes:

    @jax.jit
    def computeTrialMeansAndVars(vMean, vCov, Kzz, Kzz_inv, Ktz, KttDiag):
        # vMean \in n_latents x n_ind_points
        # vCov \in n_latents x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_ind_points x n_ind_points
        # Ktz \in n_latents x n_spikes[r] x n_ind_points
        # KttDiag \in Real
        # return (n_spikes_r x n_latents, n_spikes_r x n_latents)

        def computeLatentMeansAndVars(vMean, vCov, Kzz, Kzz_inv, Ktz, KttDiag):
            Akr = jax.scipy.linalg.cho_solve((Kzz_inv[:, :], True), vMean)
            qKMu = jnp.squeeze(jnp.matmul(Ktz[:, :], Akr))

            # Bfk \in n_ind_points x n_spikes_r[r]
            Bfk = jax.scipy.linalg.cho_solve((Kzz_inv[:, :], True),
                                             Ktz[:, :].transpose((1, 0)))

            # mm1f \in n_ind_points x n_spikes_r[r]
            diff = vCov[:, :]-Kzz[:, :]
            mm1f = jnp.matmul(diff, Bfk)

            # qKVar[r] \in nTimes[r] x n_latents
            qKVar = KttDiag + jnp.sum(a=Bfk*mm1f, axis=0)

            return qKMu, qKVar

        computeLatentMeansAndVars_vmLatents = jax.vmap(
            computeLatentMeansAndVars, (0, 0, 0, 0, 0, None), (1, 1))
        qKMu, qKVar = computeLatentMeansAndVars_vmLatents(
            vMean, vCov, Kzz, Kzz_inv, Ktz, KttDiag)
        return qKMu, qKVar

    def computeMeansAndVars(vMean, vCov, Kzz, Kzz_inv, Ktz, KttDiag=1.0):
        # vMean \in n_latents x n_trials x n_ind_points
        # vCov \in
        #  n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz \in n_latents x n_trials x n_ind_points x n_ind_points
        # Kzz_inv \in n_latents x n_trials x n_ind_points x n_ind_points
        # Ktz[r] \in n_latents x n_spikes[r] x n_ind_points
        # KttDiag \in Real
        # return qKMu[r], qKVar[r] \in n_spikes_r x n_latents

        n_trials = vMean.shape[1]
        qKMu = [[None] for tr in range(n_trials)]
        qKVar = [[None] for tr in range(n_trials)]
        for r in range(n_trials):
            # qKMu[r] \in nTimes[r] x n_latents
            qKMu[r], qKVar[r] = \
                PosteriorOnLatentsSpikes.computeTrialMeansAndVars(
                    vMean=vMean[:, r, :],
                    vCov=vCov[:, r, :, :],
                    Kzz=Kzz[:, r, :, :], Kzz_inv=Kzz_inv[:, r, :, :],
                    Ktz=Ktz[r], KttDiag=KttDiag)
        return qKMu, qKVar
