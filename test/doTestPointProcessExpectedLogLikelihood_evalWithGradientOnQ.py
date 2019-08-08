
import sys
import pdb
import numpy as np
import PointProcessExpectedLogLikelihood

def main(argv):
    nquad = 50
    nquad_link = 20
    nNeurons = 50
    nTrials = 5

    np.random.seed(seed=2344)

    legQuadPoints = np.random.normal(size=nquad)
    legQuadWeights = np.random.normal(size=nquad)
    hermQuadPoints = np.random.normal(size=nquad)
    hermQuadWeights = np.random.normal(size=nquad)
    linkFunction = np.exp

    qHMeanQuad = np.random.normal(size=(nquad, nNeurons, nTrials))
    qHVarQuad = np.exp(np.random.normal(size=(nquad, nNeurons, nTrials)))
    qHMeanSpikes = np.random.normal(size=(nquad, nNeurons, nTrials))
    qHVarSpikes = np.exp(np.random.normal(size=(nquad, nNeurons, nTrials)))

    trials = np.range(nTrials)

    ppELL = PointProcessExpectedLogLikelihood(legQuadPoints=legQuadPoints, 
                                               legQuadWeights=legQuadWeights, 
                                               hermQuadPoints=hermQuadPoints, 
                                               hermQuadWeights=hermQuadWeights, 
                                               linkFunction=linkFunction)
    ppELL.evalWithGradientOnQ(qHMeanQuad, qHVarQuad, qHMeanSpikes, qHVarSpikes,
                                          trials)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
