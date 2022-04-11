
import sys
import pdb
import math
import argparse
import pickle
import configparser
import torch
import numpy as np
import plotly.io as pio
import plotly.graph_objs as go
sys.path.append("../../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory
import plot.svGPFA.plotUtilsPlotly
import lowerBoundVsOneParamUtils
import condNumberVsOneParamUtils

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
    parser.add_argument("--paramValueStart", help="Start parameter value",
                        type=float, default=0.01)
    parser.add_argument("--paramValueEnd", help="End parameters value",
                        type=float, default=2.0)
    parser.add_argument("--paramValueStep", help="Step for parameter values", type=float, default=0.01)
    parser.add_argument("--yMin", help="Minimum y value", type=float, default=-math.inf)
    parser.add_argument("--yMax", help="Minimum y value", type=float, default=+math.inf)
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
    paramValueStart = args.paramValueStart
    paramValueEnd = args.paramValueEnd
    paramValueStep = args.paramValueStep
    yMin = args.yMin
    yMax = args.yMax
    nQuad = args.nQuad

    if intermediateDesc=="None":
        modelFilename = "results/{:08d}_estimatedModel.pickle".format(estResNumber)
    else:
        modelFilename = "results/{:08d}_{:s}_estimatedModel.pickle".format(estResNumber, intermediateDesc)

    # create model
    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    model = modelRes["model"]

    indPointsLocsKMSRegEpsilon = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._epsilon
    paramUpdateFun = lowerBoundVsOneParamUtils.getParamUpdateFun(paramType=paramType)
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    conditionNumbers = np.empty(len(paramValues))
    for i in range(len(paramValues)):
        paramUpdateFun(model=model, paramValue=paramValues[i], trial=None, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
        Kzz = model._eLL._svEmbeddingAllTimes._svPosteriorOnLatents._indPointsLocsKMS._Kzz
        eValues, _ = torch.eig(Kzz[latent][trial,:,:])
        conditionNumbers[i] = eValues[:,0].max()/eValues[:,0].min()
    title = lowerBoundVsOneParamUtils.getParamTitle(paramType=paramType, trial=trial, latent=latent, neuron=neuron, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    figFilenamePattern = condNumberVsOneParamUtils.getFigFilenamePattern(prefixNumber=estResNumber, descriptor="estimatedParam", paramType=paramType, trial=trial, latent=latent, neuron=neuron, indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon, kernelParamIndex=kernelParamIndex, indPointIndex=indPointIndex, indPointIndex2=indPointIndex2)
    layout = {
        "title": title,
        "xaxis": {"title": "Parameter Value"},
        "yaxis": {"title": "Condition Number", "range": [yMin, yMax]},
    }
    data = []
    data.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": paramValues,
                "y": conditionNumbers,
                "name": "cNum trial 0",
            },
    )
    fig = go.Figure(
        data=data,
        layout=layout,
    )
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

