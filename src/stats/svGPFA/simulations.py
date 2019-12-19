
import pdb
import stats.sampler

class GPFASimulator:

    def simulate(self, nNeurons, trialsLengths, latents, C, d,
                 linkFunction, dt, latentsEpsilon=1e-5):

        """ Simulates spikes for N=nNeurons neurons and R=len(trialLengths)
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
        nTrials = len(trialsLengths)
        nLatents = len(latents)
        spikeTimes = [[] for n in range(nTrials)]
        eSim = EmbeddingSimulator(latents=latents, C=C, d=d, 
                                  latentsEpsilon=latentsEpsilon)
        sampler = stats.sampler.Sampler()
        for r in range(nTrials):
            print("Processing trial {:d}".format(r))
            spikeTimes[r] = [[] for r in range(nNeurons)]
            for n in range(nNeurons):
                print("Processing neuron {:d}".format(n))
                eFun = eSim.getEmbeddingFunctionForNeuronAndTrial(n=n, r=r)
                def intensityFun(t, linkFunction=linkFunction,
                                 embeddingFun=eFun):
                    return(linkFunction(embeddingFun(t=t)))
                spikeTimes[r][n] = \
                    sampler.sampleInhomogeneousPP_timeRescaling(
                        intensityFun=intensityFun, T=trialsLengths[r],
                        dt=dt)
        return(spikeTimes)

class EmbeddingSimulator:

    def __init__(self, latents, C, d, latentsEpsilon):
        self._latents = latents
        self._C = C
        self._d = d
        self._latentsEpsilon = latentsEpsilon

    def getEmbeddingFunctionForNeuronAndTrial(self, n, r):
        def embeddingFun(t):
            answer = 0.0
            for k in range(len(self._latents[0])):
                answer += (self._C[n, k]*
                           self._latents[r][k](t,
                                               epsilon=self._latentsEpsilon)+
                           self._d[n])
            return answer
        return embeddingFun

