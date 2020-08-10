
import pdb
import torch
import scipy.stats
import stats.pointProcess.sampler
import stats.svGPFA.kernelsMatricesStore

class BaseSimulator:

    def getCIF(self, nTrials, latentsMeans, C, d, linkFunction):
        nNeurons = C.shape[0]
        nLatents = C.shape[1]
        cifValues = [[] for n in range(nTrials)]
        for r in range(nTrials):
            embeddings = torch.matmul(C, latentsMeans[r]) + d
            cifValues[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                cifValues[r][n] = linkFunction(embeddings[n,:])
        return(cifValues)

    def simulate(self, cifTrialsTimes, cifValues):
        nTrials = len(cifValues)
        nNeurons = len(cifValues[0])
        spikesTimes = [[] for n in range(nTrials)]
        sampler = stats.pointProcess.sampler.Sampler()
        for r in range(nTrials):
            spikesTimes[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                print("Processing trial {:d} and neuron {:d}".format(r, n))
                spikesTimes[r][n] = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(cifTimes=cifTrialsTimes[r], cifValues=cifValues[r][n], T=cifTrialsTimes[r].max()), device=cifTrialsTimes[r].device)
        return(spikesTimes)

class GPFASimulator(BaseSimulator):
    def getLatentsSamplesMeansAndSTDs(self, nTrials, meansFuncs, kernels, trialsTimes, regularizationEpsilon, dtype):
        nLatents = len(kernels)
        latentsSamples = [[] for r in range(nTrials)]
        latentsMeans = [[] for r in range(nTrials)]
        latentsSTDs = [[] for r in range(nTrials)]

        for r in range(nTrials):
            latentsSamples[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
            latentsMeans[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
            latentsSTDs[r] = torch.empty((nLatents, len(trialsTimes[r])), dtype=dtype)
            for k in range(nLatents):
                print("Processing trial {:d} and latent {:d}".format(r, k))
                gp = stats.gaussianProcesses.eval.GaussianProcess(mean=meansFuncs[k], kernel=kernels[k])
                sample, mean, cov  = gp.eval(t=trialsTimes[r], regularization=regularizationEpsilon)
                latentsSamples[r][k,:] = sample
                latentsMeans[r][k,:] = mean
                latentsSTDs[r][k,:] = torch.diag(cov).sqrt()
        return latentsSamples, latentsMeans, latentsSTDs


class GPFAwithIndPointsSimulator(BaseSimulator):
#     def _getMeansAtIndPointsLocs(self, meansFuncs, indPointsLocs):
#         nLatents = len(meansFuncs)
#         nTrials = len(indPointsLocs)
#         meansZ = [[] for r in range(nTrials)]
# 
#         for r in range(nTrials):
#             meansZ[r] = [[] for k in range(nTrials)]
#             for k in range(nLatents):
#                 meansZ[r][k] = meansFuncs[k](indPointsLocs[k][r,:,0])
#         return meansZ

    def _getIndPoints(self, meansZ, Kzz):
        nLatents = len(Kzz)
        nTrials = Kzz[0].shape[0]
        indPoints = [[] for r in range(nTrials)]

        for r in range(nTrials):
            indPoints[r] = [[] for k in range(nTrials)]
            for k in range(nLatents):
                mn = scipy.stats.multivariate_normal(mean=meansZ[r][k], cov=Kzz[k][r,:,:])
                indPoints[r][k] = torch.from_numpy(mn.rvs()).unsqueeze(1)
        return indPoints

#     def getLatentsSamplesMeansAndSTDs(self, indPointsMeans, kernels, indPointsLocs, trialsTimes, regularizationEpsilon, dtype):
#         nLatents = len(kernels)
#         nTrials = len(indPointsLocs)
# 
#         indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
#         indPointsLocsKMS.setKernels(kernels=kernels)
#         indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
#         indPointsLocsKMS.setEpsilon(epsilon=regularizationEpsilon)
#         indPointsLocsKMS.buildKernelsMatrices()
#         Kzz = indPointsLocsKMS.getKzz()
#         KzzChol = indPointsLocsKMS.getKzzChol()
# 
#         indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
#         indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
#         indPointsLocsAndAllTimesKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
#         indPointsLocsAndAllTimesKMS.setTimes(times=trialsTimes)
#         indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
#         Ktz = indPointsLocsAndAllTimesKMS.getKtz()
# 
#         latentsSamples = [[] for r in range(nTrials)]
#         latentsMeans = [[] for r in range(nTrials)]
#         latentsSTDs = [[] for r in range(nTrials)]
#         for r in range(nTrials):
#             latentsSamples[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
#             latentsMeans[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
#             latentsSTDs[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
#             for k in range(nLatents):
#                 Ak = torch.cholesky_solve(indPointsMeans[r][k].unsqueeze(1), KzzChol[k][r,:,:])
#                 mean = torch.matmul(Ktz[k][r,:,:], Ak).squeeze()
#                 cov = kernels[k].buildKernelMatrix(trialsTimes[r,:,0])
#                 cov += torch.eye(cov.shape[0])*regularizationEpsilon
#                 std = torch.diag(cov).sqrt()
#                 mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
#                 sample = torch.from_numpy(mn.rvs())
#                 latentsSamples[r][k,:] = sample
#                 latentsMeans[r][k,:] = mean
#                 latentsSTDs[r][k,:] = std
#         return latentsSamples, latentsMeans, latentsSTDs, KzzChol

    def getLatentsMeans(self, indPointsMeans, kernels, indPointsLocs, trialsTimes, indPointsLocsKMSEpsilon, dtype):
        nLatents = len(kernels)
        nTrials = len(indPointsLocs)

        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
        indPointsLocsKMS.setKernels(kernels=kernels)
        indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
        indPointsLocsKMS.setEpsilon(epsilon=indPointsLocsKMSEpsilon)
        indPointsLocsKMS.buildKernelsMatrices()
        Kzz = indPointsLocsKMS.getKzz()
        KzzChol = indPointsLocsKMS.getKzzChol()

        indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
        indPointsLocsAndAllTimesKMS.setTimes(times=trialsTimes)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()

        latentsMeans = [[] for r in range(nTrials)]
        for r in range(nTrials):
            latentsMeans[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
            for k in range(nLatents):
                Ak = torch.cholesky_solve(indPointsMeans[r][k].unsqueeze(1), KzzChol[k][r,:,:])
                latentsMeans[r][k,:] = torch.matmul(Ktz[k][r,:,:], Ak).squeeze()
        return latentsMeans, KzzChol

