
import sys
import os
import pdb
from scipy.io import loadmat
from kernelMatricesStore import IndPointsLocsAndAssocTimesKMS
from expectedLogLikelihood import PointProcessELLExpLink
from svEmbedding import LinearSVEmbeddingAssocTimes
from svPosteriorOnLatents import SVPosteriorOnLatentsAssocTimes

def main(argv):
    yNonStackedFilename = "data/YNonStacked.mat"

    mat = loadmat(yNonStackedFilename)
    nTrials = len(mat['YNonStacked'])
    indPointsLocsAndAssociatedTimesKMS = IndPointsLocsAndAssocTimesKMS()
    svPosteriorOnLatentsAssocTimes = SVPosteriorOnLatentsAssocTimes(
        svPosteriorOnIndPoints=None, indPointsLocsKMS=None, 
        indPointsLocsAndTimesKMS=indPointsLocsAndAssociatedTimesKMS)
    linearSVEmbeddingAssocTimes = LinearSVEmbeddingAssocTimes(
        svPosteriorOnLatents=svPosteriorOnLatentsAssocTimes)
    pointProcessELLExpLink = PointProcessELLExpLink(
        svEmbeddingAllTimes=None, 
        svEmbeddingForAllTimes=None,
        svEmbeddingForAssocTimes=linearSVEmbeddingAssocTimes)
    pointProcessELLExpLink.setMeasurements(measurements=mat['YNonStacked'])

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
