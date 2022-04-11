
import sys
import pdb
import argparse
import pickle
import configparser
import torch
import numpy as np
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtilsPlotly

def getReferenceParams(model, latent):
    kernelsParams = model.getKernelsParams()
    refParams = kernelsParams[latent]
    refParams = [refParams[i].clone() for i in range(len(refParams))]
    return refParams

def updateKernelParams(model, period, lengthscale, latent):
        kernelsParams = model.getKernelsParams()
        kernelsParams[latent][0] = lengthscale
        kernelsParams[latent][1] = period
        model.buildKernelsMatrices()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--latent", help="Parameter latent number", type=int, default=0)
    parser.add_argument("--lowerBoundQuantile", help="Quantile of the smallest lower bount to plot", type=float, default=0.5)
    parser.add_argument("--lengthscaleValueStart", help="Start value for lengthscale parameter", type=float, default=1.25)
    parser.add_argument("--lengthscaleValueEnd", help="End value for lengthscale parameter", type=float, default=3.25)
    parser.add_argument("--lengthscaleValueStep", help="Step size for lengthscale parameter", type=float, default=0.05)
    parser.add_argument("--periodValueStart", help="Start value for period parameter", type=float, default=2.0)
    parser.add_argument("--periodValueEnd", help="End value for period parameter", type=float, default=8.0)
    parser.add_argument("--periodValueStep", help="Step size for period parameter", type=float, default=0.05)
    parser.add_argument("--zMin", help="Minimum z value", type=float, default=None)
    parser.add_argument("--zMax", help="Minimum z value", type=float, default=None)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)

    args = parser.parse_args()
    estResNumber = args.estResNumber
    latent = args.latent
    lowerBoundQuantile = args.lowerBoundQuantile
    periodValueStart = args.periodValueStart
    periodValueEnd = args.periodValueEnd
    periodValueStep = args.periodValueStep
    lengthscaleValueStart = args.lengthscaleValueStart
    lengthscaleValueEnd = args.lengthscaleValueEnd
    lengthscaleValueStep = args.lengthscaleValueStep
    zMin = args.zMin
    zMax = args.zMax

    modelFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)

    # create model
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    indPointsLocsKMSRegEpsilon = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._epsilon
    refParams = getReferenceParams(model=model, latent=latent)
    refParamsLowerBound = model.eval()
    periodValues = np.arange(periodValueStart, periodValueEnd, periodValueStep)
    lengthscaleValues = np.arange(lengthscaleValueStart, lengthscaleValueEnd, lengthscaleValueStep)
    allLowerBoundValues = []
    allLengthscaleValues = []
    allPeriodValues = []
    for i in range(len(periodValues)):
        print("Processing period {:f} ({:d}/{:d})".format(periodValues[i], i, len(periodValues)))
        for j in range(len(lengthscaleValues)):
            updateKernelParams(model=model, lengthscale=lengthscaleValues[j], period=periodValues[i], latent=latent)
            lowerBound = model.eval()
            allLowerBoundValues.append(lowerBound.item())
            allPeriodValues.append(periodValues[i])
            allLengthscaleValues.append(lengthscaleValues[j])
    title = "Kernel Periodic, Latent {:d}, Epsilon {:f}".format(latent, indPointsLocsKMSRegEpsilon)
    figFilenamePattern = "figures/{:08d}_estimatedParams_epsilon{:f}_kernel_periodic_latent{:d}.{{:s}}".format(estResNumber, indPointsLocsKMSRegEpsilon, latent)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundVsTwoParamsParam(param1Values=allPeriodValues, param2Values=allLengthscaleValues, lowerBoundValues=allLowerBoundValues, refParam1=refParams[0], refParam2=refParams[1], refParamText="Generative Value", refParamsLowerBound=refParamsLowerBound, title=title, lowerBoundQuantile=lowerBoundQuantile, param1Label="Period", param2Label="Lengthscale", lowerBoundLabel="Lower Bound", zMin=zMin, zMax=zMax)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

