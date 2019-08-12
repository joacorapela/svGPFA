
import sys
import os
import pdb
from scipy.io import loadmat

def main(argv):
    dataFilename = os.path.expanduser("~/dev/research/gatsby/svGPFA/code/test/data/predict_MultiOutputGP.mat")

    mat = loadmat(dataFilename)
    Ktz = [mat['Ktz'][i][0] for i in range(len(mat['Ktz']))]
    Kzzi = [mat['Kzzi'][i][0] for i in range(len(mat['Kzzi']))]

    q = SparseVariationalProposal()
    qHMu, qHVar = q.getMeanAndVarianceAtQuadPoints(qMu=qMu, qSigma=qSigma, C=C, d=d, Kzzi=Kzzi, Ktz=Ktz, Ktt=Ktt):

    pdb.set_trace()

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
