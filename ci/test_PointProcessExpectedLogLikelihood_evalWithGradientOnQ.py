
def test_PointProcessExpectedLogLikelihood_evalWithGradientOnQ():
    
    ppELL = PointProcessExpectedLogLikelihood(data, legQuadPoints, legQuadWeights, hermQuadPoints, hermQuadWeights, spikeTimes):
