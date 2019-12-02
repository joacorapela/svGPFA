
class GPFASimulator:

    def simulate(self, nNeurons, trialsLengths, latents, C, d,
                 linkFunction, dt=.03):
        nNeurons = C.shape[0]
        nTrials = len(trialsLengths)
        nLatents = len(latents)
        spikeTimes = [[] for n in range(nNeurons)]
        eSim = EmbeddingSimulator(latents=latents, C=C, d=d, 
                                  linkFunction=linkFunction)
        for n in range(nNeurons):
            eFun = eSim.getEmbeddingFunctionForNeuron(n=n)
            def intensityFunction(t, linkFunction=linkFunction, 
                                  embeddingFunction=eFun):
                return(linkFunction(embeddingFunction(t=t)))
            spikeTimes[n] = [[] for r in range(nTrials)]
            for r in nTrials:
                spikeTimes[n][r] =  \
                    probs.sampler.sampleInhomogeneousPP_timeRescaling(
                        intensityFunction=intensityFunction, 
                        T=trialsLengths[r], dt=dt)
        return(spikeTimes)

class EmbeddingSimulator:

    def __init__(self, latents, C, d):
        self._latents = latents
        self._C = C
        self._d = d

    def getEmbeddingFunctionForNeuron(n):
        def embeddingFunction(t): 
            answer = 0.0
            for k in range(len(latents)):
                answer += self._C[n,k]*self._latents[k](t)+self._d[n]
            return answer
        return embeddingFunction

