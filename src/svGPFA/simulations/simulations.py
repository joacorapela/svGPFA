
import pdb
import warnings
import torch
import scipy.stats
import stats.pointProcess.sampling
import stats.svGPFA.kernelsMatricesStore

class BaseSimulator:

    def getCIF(self, nTrials, latentsSamples, C, d, linkFunction):
        nNeurons = C.shape[0]
        nLatents = C.shape[1]
        cifValues = [[] for n in range(nTrials)]
        for r in range(nTrials):
            embeddings = torch.matmul(C, latentsSamples[r]) + d
            cifValues[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                cifValues[r][n] = linkFunction(embeddings[n,:])
        return(cifValues)

    def simulate(self, cifTrialsTimes, cifValues,
                 sampling_func=
                  stats.pointProcess.sampling.sampleInhomogeneousPP_thinning):
        nTrials = len(cifValues)
        nNeurons = len(cifValues[0])
        spikesTimes = [[] for n in range(nTrials)]
        for r in range(nTrials):
            spikesTimes[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                print("Processing trial {:d} and neuron {:d}".format(r, n))
                spikesTimes[r][n] = torch.tensor(
                    sampling_func(CIF_times=cifTrialsTimes[r],
                                  CIF_values=cifValues[r][n])["inhomogeneous"])
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

    def getLatentsSamplesMeansAndSTDs(self, indPointsMeans, kernels,
                                      indPointsLocs, trialsTimes,
                                      indPointsLocsKMSRegEpsilon,
                                      latentsCovRegEpsilon, dtype):
        nLatents = len(kernels)
        nTrials = indPointsLocs[0].shape[0]

        indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS_Chol()
        indPointsLocsKMS.setKernels(kernels=kernels)
        indPointsLocsKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
        indPointsLocsKMS.setEpsilon(epsilon=indPointsLocsKMSRegEpsilon)
        indPointsLocsKMS.buildKernelsMatrices()
        Kzz = indPointsLocsKMS.getKzz()
        # pdb.set_trace()
        # begin debug
        condNumberThr = 1e+6
        eigRes = torch.eig(Kzz[0][0,:,:], eigenvectors=True)
        if any(eigRes.eigenvalues[:,1]>0):
            warnings.warn("Some eigenvalues of Kzz are imaginary")
        sortedEigVals = eigRes.eigenvalues[:,0].sort(descending=True).values
        condNumber = sortedEigVals[0]/sortedEigVals[-1]
        if condNumber>condNumberThr:
            warnings.warn("Poorly conditioned Kzz (condition number={:.02f})".format(condNumber))
        # end debug

        indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
        indPointsLocsAndAllTimesKMS.setKernels(kernels=kernels)
        indPointsLocsAndAllTimesKMS.setIndPointsLocs(indPointsLocs=indPointsLocs)
        indPointsLocsAndAllTimesKMS.setTimes(times=trialsTimes)
        indPointsLocsAndAllTimesKMS.buildKernelsMatrices()
        Ktz = indPointsLocsAndAllTimesKMS.getKtz()

        latentsSamples = [[] for r in range(nTrials)]
        latentsMeans = [[] for r in range(nTrials)]
        latentsSTDs = [[] for r in range(nTrials)]
        for r in range(nTrials):
            latentsSamples[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
            latentsMeans[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
            latentsSTDs[r] = torch.empty((nLatents, trialsTimes.shape[1]), dtype=torch.double)
            for k in range(nLatents):
                # mean = torch.matmul(Ktz[k][r,:,:], torch.cholesky_solve(indPointsMeans[r][k], KzzChol[k][r,:,:])).squeeze()
                tmp = indPointsLocsKMS.solveForLatentAndTrial(
                    input=indPointsMeans[r][k], latentIndex=k, trialIndex=r)
                mean = torch.matmul(Ktz[k][r,:,:], tmp).squeeze()
                cov = kernels[k].buildKernelMatrix(trialsTimes[r,:,0])
                cov += torch.eye(cov.shape[0])*latentsCovRegEpsilon
                std = torch.diag(cov).sqrt()
                mn = scipy.stats.multivariate_normal(mean=mean, cov=cov)
                sample = torch.from_numpy(mn.rvs())
                latentsSamples[r][k,:] = sample
                latentsMeans[r][k,:] = mean
                latentsSTDs[r][k,:] = std
        return latentsSamples, latentsMeans, latentsSTDs, Kzz

