
import stats.svGPFA.kernelsMatricesStore
import stats.svGPFA.svPosteriorOnIndPoints
import stats.svGPFA.svPosteriorOnLatents
import stats.svGPFA.svEmbedding
import stats.svGPFA.expectedLogLikelihood
import stats.svGPFA.klDivergence
import stats.svGPFA.svLowerBound

#:
PointProcess = 0
#:
Poisson = 1
#:
Gaussian = 2


#:
LinearEmbedding = 100

#:
ExponentialLink = 1000
#:
NonExponentialLink = 1001

class SVGPFAModelFactory:

    @staticmethod
    def buildModel(conditionalDist, linkFunction, embeddingType, kernels, paramsLogPriors):

        if conditionalDist==PointProcess:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    qU = stats.svGPFA.svPosteriorOnIndPoints.SVPosteriorOnIndPoints()
                    indPointsLocsKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsKMS()
                    indPointsLocsAndAllTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = stats.svGPFA.kernelsMatricesStore.IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = stats.svGPFA.svPosteriorOnLatents.SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = stats.svGPFA.svEmbedding.LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = stats.svGPFA.expectedLogLikelihood.PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = stats.svGPFA.klDivergence.KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = stats.svGPFA.svLowerBound.SVLowerBound(eLL=eLL, klDiv=klDiv, paramsLogPriors=paramsLogPriors)
                    svlb.setKernels(kernels=kernels)
                else:
                    raise ValueError("Invalid linkFunction=%s"%
                                     repr(linkFunction))
            else:
                raise ValueError("Invalid embeddingType=%s"%
                                 repr(embeddingType))
        else:
            raise ValueError("Invalid conditionalDist=%s"%
                             repr(conditionalDist))

        return svlb
