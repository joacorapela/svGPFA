
import pdb
import torch
import stats.pointProcess.sampler

class GPFASimulator:

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

