
import pdb
import probs.sampler

class GPFASimulator:

    def simulate(self, nNeurons, trialsLengths, latents, C, d,
                 linkFunction, dt):
        '''
        Simulates spikes for N=nNeurons neurons and R=len(trialLengths) trials
        using K=len(latents) per trial.

        Parameters
        ----------

        nNeurons: int
                  number of neurons to simulate.
        trialsLengths: numpy array \in R^R
                       trialsLengths[r] is the duration, T_r, of the rth trial
        latents: list of length K
            len(latents[k])=R and contains kth latent processes (i.e., Gaussian
            processes) for all R trials.
        C: numpy ndarray \in R^{N\times K}
        d: numpy array \in R^N
        linkFunction: function
                      function to map embedding values to point-process
                      intensity values.

        Returns
        -------

        list[n][r]
            containing a list of spike times for neuron n in trial r

        ''' 
        nNeurons = C.shape[0]
        nTrials = len(trialsLengths)
        nLatents = len(latents)
        spikeTimes = [[] for n in range(nTrials)]
        eSim = EmbeddingSimulator(latents=latents, C=C, d=d)
        sampler = probs.sampler.Sampler()
        for r in range(nTrials):
            print("Processing trial {:d}".format(r))
            spikeTimes[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                print("Processing neuron {:d}".format(n))
                eFun = eSim.getEmbeddingFunctionForNeuronAndTrial(n=n, r=r)
                def intensityFun(t, linkFunction=linkFunction,
                                 embeddingFun=eFun):
                    return(linkFunction(embeddingFun(t=t)))
                spikeTimes[r][n] = sampler.sampleInhomogeneousPP_timeRescaling(
                    intensityFun=intensityFun, T=trialsLengths[r],
                    dt=dt)
        return(spikeTimes)

class EmbeddingSimulator:

    def __init__(self, latents, C, d):
        self._latents = latents
        self._C = C
        self._d = d

    def getEmbeddingFunctionForNeuronAndTrial(self, n, r):
        def embeddingFun(t):
            answer = 0.0
            for k in range(len(self._latents[0])):
                answer += self._C[n,k]*self._latents[r][k](t)+self._d[n]
            return answer
        return embeddingFun

