
import pdb
import torch
import stats.pointProcess.sampler

class GPFASimulator:

    def simulate(self, trialsTimes, latentsSamples, C, d, linkFunction):

        """ Simulates spikes for N=C.shape[0] neurons and R=len(trialLengths)
        trials using K=len(latents) latents.

        :param: nNeurons: number of neurons to simulate
        :type: nNeurons: int

        :param: trialsLengths: trialsLengths[r] is the duration of the rth trial
        :type: trialsLengths: numpy array.

        :param: latents: len(latents[k])=R and contains kth latent processes (i.e., Gaussian processes) for all R trials.
        :type: latents: list of length K

        :param: C: matrix mapping latent to neural activity
        :type: C: numpy ndarray :math:`\in R^{N \\times K}`

        :param: d: constant in mapping latent to neural activity
        :type: d: numpy array :math:`\in R^N`

        :params: linkFunction: function mapping embedding values to point-process intensity values.
        :type: linkFunction: function

        :return: list where list[n][r] contains the of spike times for neuron n in trial r

        """
        nNeurons = C.shape[0]
        nLatents = C.shape[1]
        nTrials = len(trialsTimes)
        spikesTimes = [[] for n in range(nTrials)]
        intensityValues = [[] for n in range(nTrials)]
        sampler = stats.pointProcess.sampler.Sampler()
        for r in range(nTrials):
            embeddings = torch.matmul(C, latentsSamples[r]) + d
            print("Processing trial {:d}".format(r))
            spikesTimes[r] = [[] for r in range(nNeurons)]
            intensityValues[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                print("Processing neuron {:d}".format(n))
                intensityValues[r][n] = linkFunction(embeddings[n,:])
                spikesTimes[r][n] = torch.tensor(sampler.sampleInhomogeneousPP_timeRescaling(intensityTimes=trialsTimes[r], intensityValues=intensityValues[r][n], T=trialsTimes[r].max()), device=C.device)
                # pdb.set_trace()
        answer = {"spikesTimes": spikesTimes, "intensityTimes": trialsTimes, "intensityValues": intensityValues}
        return(answer)

