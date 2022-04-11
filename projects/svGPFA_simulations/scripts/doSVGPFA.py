
import sys
from ConfigParser import SafeConfigParser

def main(argv):
    configFilename = argv[1]

    parser = SafeConfigParser()
    parser.read(configFilename)

    g = eval(parser.get('sparseEstimation', 'linkFunction'))
    k = eval(parser.get('sparseEstimation', 'kernelFunction'))
    y = np.load(parser.get('data', 'yFilename'))
    z0 = np.load(parser.get('sparseEstimation', 'inducingPointsFilename'))
    nHermQuadPoints = parser.get('sparseEstimation', 'nQuadHermPoints')
    nLegQuadPoints = parser.get('sparseEstimation', 'nQuadLegPoints')
    qMu0 = np.load(parser.get('initialValues', 'qMu0Filename'))
    qSigma0 = np.load(parser.get('initialValues', 'qSigma0Filename'))
    C0 = np.load(parser.get('initialValues', 'C0Filename'))
    d0 = np.load(parser.get('initialValues', 'd0Filename'))
    kernelParams0 = np.fromstring(parser.get('initialValues', 'kernelParams0'),
                                   dtype='double', sep=',')
    maxIter = int(parser.get('sparseEstimation', 'maxIter')
    tol = float(parser.get('sparseEstimation', 'tol')

    hermQxx, hermQww = np.polynomial.hermite.hermgauss(deg=nHermQuadPoints)
    legQxx, legQww = np.polynomial.legendre.leggauss(deg=nLegQuadPoints)
    cMS = PointProcessCovarianceMatricesStore(hermQxx=hermQxx, hermQww=hermQww)
    qU = InducingPointsPrior(qMu=qMu0, qSVec=qSVec0, qSDiag=qSDiag0)
    qH = PointProcessApproxPosteriorForH(C=C0, 
                                          d=d0,
                                          linkFunction=g, 
                                          inducingPointsPrior=qU,
                                          covMatricesStore=cMS)
    eLL = PointProcessExpectedLogLikelihood(approxPosteriorForH=qH, 
                                             legQuadPoints=legQuadPoints, 
                                             legQuadWeights=legQuadWeights, 
                                             hermQuadPoints, hermQuadWeights, 
                                             linkFunction=g)
    kLDivergence = KLDivergence(inucingPointsPrior=qU, covMatricesStore=cMS)
    lowerBound = SparseVariationalLowerbound(eLL=eLL, klDivergece=klDivergence)
    svEM = SparseVariationalEM(lowerBound=lowerBound)

    qMu, qSigma, C, d, kernelsHyperParams, z = svEM.maximize()

if __name__=="__main__":
    main(sys.argv)
