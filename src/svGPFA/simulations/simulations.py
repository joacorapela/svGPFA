
import warnings
import torch
import scipy.stats
import gcnu_common.stats.gaussianProcesses.eval
import gcnu_common.stats.pointProcesses.sampling
from ..stats import kernelsMatricesStore


class BaseSimulator:

    def getCIF(self, n_trials, latents_samples, C, d, link_function):
        n_neurons = C.shape[0]
        cif_values = [[] for n in range(n_trials)]
        for r in range(n_trials):
            embeddings = torch.matmul(C, latents_samples[r]) + d
            cif_values[r] = [link_function(embeddings[n, :])
                             for n in range(n_neurons)]
        return(cif_values)

    def simulate(self, cif_trials_times, cif_values,
                 sampling_func=gcnu_common.stats.pointProcesses.sampling.sampleInhomogeneousPP_thinning):
        n_trials = len(cif_values)
        n_neurons = len(cif_values[0])
        spikes_times = [[] for n in range(n_trials)]
        for r in range(n_trials):
            spikes_times[r] = [[] for r in range(n_neurons)]
            for n in range(n_neurons):
                print("Processing trial {:d} and neuron {:d}".format(r, n))
                spikes_times[r][n] = torch.tensor(
                    sampling_func(CIF_times=cif_trials_times[r],
                                  CIF_values=cif_values[r][n])["inhomogeneous"])
        return(spikes_times)


class GPFASimulator(BaseSimulator):
    def getLatentsSamplesMeansAndSTDs(self, n_trials, means_funcs, kernels,
                                      trials_times, prior_cov_reg_param,
                                      dtype):
        n_latents = len(kernels)
        latents_samples = [[] for r in range(n_trials)]
        latents_means = [[] for r in range(n_trials)]
        latents_STDs = [[] for r in range(n_trials)]

        for r in range(n_trials):
            latents_samples[r] = torch.empty((n_latents, len(trials_times[r])),
                                             dtype=dtype)
            latents_means[r] = torch.empty((n_latents, len(trials_times[r])),
                                           dtype=dtype)
            latents_STDs[r] = torch.empty((n_latents, len(trials_times[r])),
                                          dtype=dtype)
            for k in range(n_latents):
                print("Processing trial {:d} and latent {:d}".format(r, k))
                gp = gcnu_common.stats.gaussianProcesses.eval.GaussianProcess(
                    mean=means_funcs[k], kernel=kernels[k])
                sample, mean, cov = gp.eval(
                    t=trials_times[r], regularization=prior_cov_reg_param)
                latents_samples[r][k, :] = sample
                latents_means[r][k, :] = mean
                latents_STDs[r][k, :] = torch.diag(cov).sqrt()
        return latents_samples, latents_means, latents_STDs


class GPFAwithIndPointsSimulator(BaseSimulator):

    def getLatentsSamplesMeansAndSTDs(self, var_mean, var_cov, kernels,
                                      ind_points_locs, trials_times,
                                      prior_cov_reg_param,
                                      latents_cov_reg_param, dtype):
        n_latents = len(kernels)
        n_trials = ind_points_locs[0].shape[0]

        indPointsLocsKMS = kernelsMatricesStore.IndPointsLocsKMS_Chol()
        indPointsLocsKMS.setKernels(kernels=kernels)
        indPointsLocsKMS.setIndPointsLocs(ind_points_locs=ind_points_locs)
        indPointsLocsKMS.setRegParam(reg_param=prior_cov_reg_param)
        indPointsLocsKMS.buildKernelsMatrices()
        Kzz = indPointsLocsKMS.getKzz()
        condNumberThr = 1e+6
        eigRes = torch.eig(Kzz[0][0, :, :], eigenvectors=True)
        if any(eigRes.eigenvalues[:, 1] > 0):
            warnings.warn("Some eigenvalues of Kzz are imaginary")
        sortedEigVals = eigRes.eigenvalues[:, 0].sort(descending=True).values
        condNumber = sortedEigVals[0]/sortedEigVals[-1]
        if condNumber > condNumberThr:
            warnings.warn("Poorly conditioned Kzz "
                          f"(condition number={condNumber}")

        indPointsLocsAndAllTimesKMS = \
            kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(
            ind_points_locs=ind_points_locs)
        indPointsLocsAndAllTimesKMS.setTimes(times=trials_times)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()

        latents_samples = [[] for r in range(n_trials)]
        latents_means = [[] for r in range(n_trials)]
        latents_STDs = [[] for r in range(n_trials)]
        for r in range(n_trials):
            latents_samples[r] = torch.empty((n_latents,
                                              trials_times.shape[1]),
                                             dtype=torch.double)
            latents_means[r] = torch.empty((n_latents, trials_times.shape[1]),
                                           dtype=torch.double)
            latents_STDs[r] = torch.empty((n_latents, trials_times.shape[1]),
                                          dtype=torch.double)
            for k in range(n_latents):
                # first let's compute the variational mean
                tmp = indPointsLocsKMS.solveForLatentAndTrial(
                    input=var_mean[r][k], latentIndex=k, trialIndex=r)
                mean = torch.matmul(Ktz[k][r, :, :], tmp).squeeze()
                # done with the variational mean

                # not let's compute the variational covariance
                # cov = Ktt + Ktz Kzz^{-1} (Sk-Kzz) Kzz^{-1} Kst
                # cov = Ktt + Ktz C
                # A = solve(Kzz, Kzt)
                # B = (Sk-Kzz) A
                # C = solze(Kzz, B)
                A = indPointsLocsKMS.solveForLatentAndTrial(
                    input=Ktz[k][r, :, :].T, latentIndex=k, trialIndex=r)
                B = torch.matmul(var_cov[r][k]-Kzz[k][r, :, :], A)
                C = indPointsLocsKMS.solveForLatentAndTrial(
                    input=B, latentIndex=k, trialIndex=r)
                cov = kernels[k].buildKernelMatrix(trials_times[r, :, 0]) + \
                    torch.matmul(Ktz[k][r, :, :], C)
                # done with the variational cov

                std = torch.diag(cov).sqrt()
                cov += torch.eye(cov.shape[0])*latents_cov_reg_param
                mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
                sample = torch.from_numpy(mn.rvs())
                latents_samples[r][k, :] = sample
                latents_means[r][k, :] = mean
                latents_STDs[r][k, :] = std
        return latents_samples, latents_means, latents_STDs, Kzz
