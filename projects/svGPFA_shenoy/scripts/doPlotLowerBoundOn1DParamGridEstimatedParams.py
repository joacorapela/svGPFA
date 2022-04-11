
import sys
import pdb
import math
import argparse
import configparser
import pickle
import numpy as np
import torch
import plotly.io as pio
sys.path.append("../../src")
import utils.svGPFA.configUtils
import plot.svGPFA.plotUtilsPlotly
import lowerBoundVsOneParamUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("paramType", help="Parameter type: indPointsPosteriorMean, indPointsPosteriorCov, indPointsLocs, kernel, embeddingC, embeddingD")
    parser.add_argument("--intermediateDesc", help="Descriptor of the intermediate model estimate to plot", type=str, default="None")
    parser.add_argument("--trial", help="Parameter trial number", type=int, default=0)
    parser.add_argument("--latent", help="Parameter latent number", type=int, default=0)
    parser.add_argument("--neuron", help="Parameter neuron number", type=int, default=0)
    parser.add_argument("--kernelParamIndex", help="Kernel parameter index", type=int, default=0)
    parser.add_argument("--indPointIndex", help="Parameter inducing point index", type=int, default=0)
    parser.add_argument("--indPointIndex2", help="Parameter inducing point index2 (used for inducing points covariance parameters)", type=int, default=0)
    parser.add_argument("--scaledParamValueStart", help="Start scaled parameter value", type=float, default=0.01)
    parser.add_argument("--scaledParamValueEnd", help="End scaled parameters value", type=float, default=2.00)
    parser.add_argument("--scaledParamValueStep", help="Step for scaled parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
    parser.add_argument("--percMargin", help="Percentage value for margin=perecMargin*max(abs(yMin), abs(yMax))", type=float, default=0.1)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)

    args = parser.parse_args()
    estResNumber = args.estResNumber
    paramType = args.paramType
    intermediateDesc = args.intermediateDesc
    trial = args.trial
    latent = args.latent
    neuron = args.neuron
    kernelParamIndex = args.kernelParamIndex
    indPointIndex = args.indPointIndex
    indPointIndex2 = args.indPointIndex2
    scaledParamValueStart = args.scaledParamValueStart
    scaledParamValueEnd = args.scaledParamValueEnd
    scaledParamValueStep = args.scaledParamValueStep
    yMin = args.yMin
    yMax = args.yMax
    percMargin = args.percMargin
    nQuad = args.nQuad

    if paramType=="kernel":
        estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estResNumber)
        estMetaDataConfig = configparser.ConfigParser()
        estMetaDataConfig.read(estMetaDataFilename)
        estInitNumber = int(estMetaDataConfig["estimation_params"]["estInitNumber"])
        estInitConfig = configparser.ConfigParser()
        estInitFilename = "data/{:08d}_estimation_metaData.ini".format(estInitNumber)
        estInitConfig.read(estInitFilename)
        if kernelParamIndex==0:
            refParam0 = float(estInitConfig["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(latent)])
            refParam0Scale = float(estInitConfig["kernel_params"]["kLengthscaleScaleLatent{:d}".format(latent)])
        elif kernelParamIndex==1:
            refParam0 = float(estInitConfig["kernel_params"]["kLengthscaleScaledValueLatent{:d}".format(latent)])
            refParam0Scale = float(estInitConfig["kernel_params"]["kLengthscaleScaleLatent{:d}".format(latent)])
        else:
            raise NotImplementedError("Currently only kernelParamIndex=0 or 1 are supported")
    else:
        raise NotImplementedError("Currently only paramType=kernel is supported")

    if intermediateDesc=="None":
        modelFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    else:
        modelFilename = "results/{:08d}_{:s}_estimatedModel.pickle".format(estResNumber, intermediateDesc)

    # create model
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    refParams = [refParam0]
    indPointsLocsKMSRegEpsilon = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._epsilon
    refParams.append(lowerBoundVsOneParamUtils.getReferenceParam(paramType=paramType, model=model, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2).detach().numpy()/refParam0Scale)
    paramUpdateFun = lowerBoundVsOneParamUtils.getParamUpdateFun(paramType=paramType)
    scaledParamValues = np.arange(scaledParamValueStart, scaledParamValueEnd, scaledParamValueStep)
    lowerBoundValues = np.empty(scaledParamValues.shape)
    for i in range(len(scaledParamValues)):
        with torch.no_grad():
            paramUpdateFun(model=model, paramValue=scaledParamValues[i]*refParam0Scale, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
#         if scaledParamValues[i]>=6.62:
#             pdb.set_trace()
            pdb.set_trace()
            lowerBoundValues[i] = model.eval()
    title = lowerBoundVsOneParamUtils.getParamTitle(paramType=paramType, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    if intermediateDesc=="None":
        figDesc = "lowerBoundVs1DParam_estimatedParams"
    else:
        figDesc = "lowerBoundVs1DParam_{:s}_estimatedParams".format(intermediateDesc)
    figFilenamePattern = lowerBoundVsOneParamUtils.getFigFilenamePattern(prefixNumber=estResNumber, descriptor=figDesc, paramType=paramType, trial=trial, latent=latent, neuron=neuron, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
    fig = plot.svGPFA.plotUtilsPlotly.getPlotLowerBoundVsOneParam(paramValues=scaledParamValues, lowerBoundValues=lowerBoundValues, refParams=refParams, title=title, yMin=yMin, yMax=yMax, lowerBoundLineColor="red", refParamsLineColors=["red", "magenta"], percMargin=percMargin)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

