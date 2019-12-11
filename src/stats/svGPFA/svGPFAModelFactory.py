
from .kernelMatricesStore import IndPointsLocsKMS, IndPointsLocsAndAllTimesKMS,\
                                IndPointsLocsAndAssocTimesKMS
from .svPosteriorOnIndPoints import SVPosteriorOnIndPoints
from .svPosteriorOnLatents import SVPosteriorOnLatentsAllTimes,\
                                 SVPosteriorOnLatentsAssocTimes
from .svEmbedding import LinearSVEmbeddingAllTimes, LinearSVEmbeddingAssocTimes
from .expectedLogLikelihood import PointProcessELLExpLink
from .klDivergence import KLDivergence
from .svLowerBound import SVLowerBound

PointProcess = 0
Poisson = 1

LinearEmbedding = 100

ExponentialLink = 1000
OtherLink = 1001

class SVGPFAModelFactory:

    @staticmethod
    def buildModel(conditionalDist, linkFunction, embeddingType, kernels):

        if conditionalDist==PointProcess:
            if embeddingType==LinearEmbedding:
                if linkFunction==ExponentialLink:
                    qU = SVPosteriorOnIndPoints()
                    indPointsLocsKMS = IndPointsLocsKMS()
                    indPointsLocsAndAllTimesKMS = IndPointsLocsAndAllTimesKMS()
                    indPointsLocsAndAssocTimesKMS = IndPointsLocsAndAssocTimesKMS()
                    qKAllTimes = SVPosteriorOnLatentsAllTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAllTimesKMS)
                    qKAssocTimes = SVPosteriorOnLatentsAssocTimes(
                        svPosteriorOnIndPoints=qU,
                        indPointsLocsKMS=indPointsLocsKMS,
                        indPointsLocsAndTimesKMS=indPointsLocsAndAssocTimesKMS)
                    qHAllTimes = LinearSVEmbeddingAllTimes(
                        svPosteriorOnLatents=qKAllTimes)
                    qHAssocTimes = LinearSVEmbeddingAssocTimes(
                        svPosteriorOnLatents=qKAssocTimes)
                    eLL = PointProcessELLExpLink(
                        svEmbeddingAllTimes=qHAllTimes,
                        svEmbeddingAssocTimes=qHAssocTimes)
                    klDiv = KLDivergence(indPointsLocsKMS=indPointsLocsKMS,
                                         svPosteriorOnIndPoints=qU)
                    svlb = SVLowerBound(eLL=eLL, klDiv=klDiv)
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
