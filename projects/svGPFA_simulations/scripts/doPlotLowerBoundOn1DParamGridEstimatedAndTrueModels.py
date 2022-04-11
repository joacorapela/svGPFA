
import sys
import pdb
import math
import argparse
import scipy.io
import pickle
import configparser
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.subplots
import plotly.io as pio
sys.path.append("../src")
import utils.svGPFA.initUtils
import utils.svGPFA.configUtils
import utils.svGPFA.miscUtils
import stats.svGPFA.svGPFAModelFactory

def getPlotLowerBounds(paramValues, tLowerBoundValues, eLowerBoundValues, tParamValue, eParamValue, xlabel, tYMin, tYMax, eYMin, eYMax, ylabel="Lower Bound", tColor="blue", eColor="red"):
    fig = plotly.subplots.make_subplots(rows=2, cols=1, subplot_titles=("Generative model", "Estimated model"))
    tTrace = go.Scatter(
        x=paramValues,
        y=tLowerBoundValues,
        mode="lines+markers",
        name="true",
        line=dict(color=tColor),
        showlegend=False,
    )
    eTrace = go.Scatter(
        x=paramValues,
        y=eLowerBoundValues,
        mode="lines+markers",
        name="estimated",
        line=dict(color=eColor),
        showlegend=False,
    )
    tVLine = dict(
        type="line",
        x0=tParamValue,
        y0=tYMin,
        x1=tParamValue,
        y1=tYMax,
        line=dict(
            color=tColor,
            width=3,
        ),
        name="generative param",
    )
    eVLine = dict(
        type="line",
        x0=eParamValue,
        y0=eYMin,
        x1=eParamValue,
        y1=eYMax,
        line=dict(
            color=eColor,
            width=3,
        ),
        name="estimated param",
    )
    fig.add_trace(tTrace, row=1, col=1)
    fig.update_xaxes(title_text=xlabel, row=1, col=1)
    if tYMin is not None and tYMax is not None:
        fig.update_yaxes(title_text=ylabel, range=[tYMin, tYMax], row=1, col=1)

    fig.add_trace(eTrace, row=2, col=1)
    fig.update_xaxes(title_text=xlabel, row=2, col=1)
    if eYMin is not None and eYMax is not None:
        fig.update_yaxes(title_text=ylabel, range=[eYMin, eYMax], row=2, col=1)

    fig.add_shape(tVLine, row=1, col=1)
    fig.add_shape(eVLine, row=2, col=1)
    return fig

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estNumber", help="estimation result number", type=int)
    parser.add_argument("--nQuad", help="Number of quadrature points", type=int, default=200)
    args = parser.parse_args()
    estNumber = args.estNumber
    nQuad = args.nQuad

    estMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(estNumber)
    modelFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    figFilenamePattern = "figures/{:08d}_lowerBoundVs{{:s}}.{{:s}}".format(estNumber)

    estMetaDataConfig = configparser.ConfigParser()
    estMetaDataConfig.read(estMetaDataFilename)
    simResNumber = int(estMetaDataConfig["simulation_params"]["simResNumber"])

    # load data and initial values
    simResConfigFilename = "results/{:08d}_simulation_metaData.ini".format(simResNumber)
    simResConfig = configparser.ConfigParser()
    simResConfig.read(simResConfigFilename)
    simInitConfigFilename = simResConfig["simulation_params"]["simInitConfigFilename"]
    simResFilename = simResConfig["simulation_results"]["simResFilename"]

    simInitConfig = configparser.ConfigParser()
    simInitConfig.read(simInitConfigFilename)
    nIndPointsPerLatent = [int(str) for str in simInitConfig["control_variables"]["nIndPointsPerLatent"][1:-1].split(",")]
    nLatents = len(nIndPointsPerLatent)
    nNeurons = int(simInitConfig["control_variables"]["nNeurons"])
    trialsLengths = [float(str) for str in simInitConfig["control_variables"]["trialsLengths"][1:-1].split(",")]
    nTrials = len(trialsLengths)
    firstIndPointLoc = float(simInitConfig["control_variables"]["firstIndPointLoc"])
    indPointsLocsKMSRegEpsilon = float(simInitConfig["control_variables"]["indPointsLocsKMSRegEpsilon"])

    with open(simResFilename, "rb") as f: simRes = pickle.load(f)
    spikesTimes = simRes["spikes"]
    KzzChol = simRes["KzzChol"]
    indPointsMeans = simRes["indPointsMeans"]
    C, d = utils.svGPFA.configUtils.getLinearEmbeddingParams(CFilename=simInitConfig["embedding_params"]["C_filename"], dFilename=simInitConfig["embedding_params"]["d_filename"])

    # patch to acommodate Lea's equal number of inducing points across trials
    qMu0 = [[] for k in range(nLatents)]
    for k in range(nLatents):
        qMu0[k] = torch.empty((nTrials, nIndPointsPerLatent[k], 1), dtype=torch.double)
        for r in range(nTrials):
            qMu0[k][r,:,:] = indPointsMeans[r][k]
    # end patch

    legQuadPoints, legQuadWeights = utils.svGPFA.miscUtils.getLegQuadPointsAndWeights(nQuad=nQuad, trialsLengths=trialsLengths)

    kernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=simInitConfig, forceUnitScale=False)
    kernelsParams0 = utils.svGPFA.initUtils.getKernelsParams0(kernels=kernels, noiseSTD=0.0)

    Z0 = utils.svGPFA.initUtils.getIndPointLocs0(nIndPointsPerLatent=nIndPointsPerLatent, trialsLengths=trialsLengths, firstIndPointLoc=firstIndPointLoc)
    srQSigma0Vecs = utils.svGPFA.initUtils.getSRQSigmaVecsFromSRMatrices(srMatrices=KzzChol)
    qUParams0 = {"qMu0": qMu0, "srQSigma0Vecs": srQSigma0Vecs}
    kmsParams0 = {"kernelsParams0": kernelsParams0,
                  "inducingPointsLocs0": Z0}
    qKParams0 = {"svPosteriorOnIndPoints": qUParams0,
                 "kernelsMatricesStore": kmsParams0}
    qHParams0 = {"C0": C, "d0": d}
    initialParams = {"svPosteriorOnLatents": qKParams0,
                     "svEmbedding": qHParams0}
    quadParams = {"legQuadPoints": legQuadPoints,
                  "legQuadWeights": legQuadWeights}

    # create model
    tModel = stats.svGPFA.svGPFAModelFactory.SVGPFAModelFactory.buildModel(
        conditionalDist=stats.svGPFA.svGPFAModelFactory.PointProcess,
        linkFunction=stats.svGPFA.svGPFAModelFactory.ExponentialLink,
        embeddingType=stats.svGPFA.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels)

    tModel.setMeasurements(measurements=spikesTimes)
    tModel.setInitialParams(initialParams=initialParams)
    tModel.setQuadParams(quadParams=quadParams)
    tModel.setIndPointsLocsKMSRegEpsilon(indPointsLocsKMSRegEpsilon=indPointsLocsKMSRegEpsilon)
    tModel.buildKernelsMatrices()


    with open(modelFilename, "rb") as f: modelRes = pickle.load(f)
    eModel = modelRes["model"]

#     tParams = tModel.getSVPosteriorOnIndPointsParams()
#     tParamValue = tParams[0][0,0,0].item()
#     eParams = eModel.getSVPosteriorOnIndPointsParams()
#     eParamValue = eParams[0][0,0,0].item()
#     paramValueStart = -3
#     paramValueEnd = 3
#     paramValueStep = .01
#     tYMin = -300000.0
#     tYMax = 100000.0
#     eYMin = -300000.0
#     eYMax = 100000.0
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     tLowerBoundValues = np.empty(paramValues.shape)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tParams[0][0,0,0] = paramValues[i]
#         tLowerBoundValues[i] = tModel.eval()
#         eParams[0][0,0,0] = paramValues[i]
#         eLowerBoundValues[i] = eModel.eval()
#     tParams[0][0,0,0] = tParamValue
#     eParams[0][0,0,0] = eParamValue
#     fig = getPlotLowerBounds(paramValues=paramValues, tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel=r"$m_k[0]$", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
#     fig.write_image(figFilenamePattern.format("mk0", "png"))
#     fig.write_html(figFilenamePattern.format("mk0", "html"))
#     pio.renderers.default = "browser"
#     fig.show()

#     tParams = tModel.getSVPosteriorOnIndPointsParams()
#     tParamValue = tParams[1][0,0,0].item()
#     eParams = eModel.getSVPosteriorOnIndPointsParams()
#     eParamValue = eParams[1][0,0,0].item()
#     paramValueStart = -3
#     paramValueEnd = 3
#     paramValueStep = .01
#     tYMin = -300000.0
#     tYMax = 100000.0
#     eYMin = -300000.0
#     eYMax = 100000.0
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     tLowerBoundValues = np.empty(paramValues.shape)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tParams[1][0,0,0] = paramValues[i]
#         tLowerBoundValues[i] = tModel.eval()
#         eParams[1][0,0,0] = paramValues[i]
#         eLowerBoundValues[i] = eModel.eval()
#     tParams[0][0,0,0] = tParamValue
#     eParams[0][0,0,0] = eParamValue
#     fig = getPlotLowerBounds(paramValues=paramValues, tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel=r"$S_k[0]$", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
#     fig.write_image(figFilenamePattern.format("Sk0", "png"))
#     fig.write_html(figFilenamePattern.format("Sk0", "html"))
#     pio.renderers.default = "browser"
#     fig.show()

#     tParams = tModel.getKernelsParams()
#     tParamValue = tParams[0][0].item()
#     eParams = eModel.getKernelsParams()
#     eParamValue = eParams[0][0].item()
#     paramValueStart = 1.0
#     paramValueEnd = 5.0
#     paramValueStep = .01
#     tYMin = None
#     tYMax = None
#     eYMin = None
#     eYMax = None
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     tLowerBoundValues = np.empty(paramValues.shape)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tParams[0][0] = paramValues[i]
#         tModel.buildKernelsMatrices()
#         tLowerBoundValues[i] = tModel.eval()
#         eParams[0][0] = paramValues[i]
#         eModel.buildKernelsMatrices()
#         eLowerBoundValues[i] = eModel.eval()
#     tParams[0][0] = tParamValue
#     eParams[0][0] = eParamValue
#     fig = getPlotLowerBounds(paramValues=paramValues, tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel="Lengthscale", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
#     fig.write_image(figFilenamePattern.format("Lengthscale", "png"))
#     fig.write_html(figFilenamePattern.format("Lengthscale", "html"))
#     pio.renderers.default = "browser"
#     fig.show()

    tParams = tModel.getKernelsParams()
    tParamValue = tParams[0][1].item()
    eParams = eModel.getKernelsParams()
    eParamValue = eParams[0][1].item()
    paramValueStart = 1.0
    paramValueEnd = 20
    paramValueStep = .01
    paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
    tLowerBoundValues = np.empty(paramValues.shape)
    eLowerBoundValues = np.empty(paramValues.shape)
    for i in range(len(paramValues)):
        tParams[0][1] = paramValues[i]
        tModel.buildKernelsMatrices()
        tLowerBoundValues[i] = tModel.eval()
        eParams[0][1] = paramValues[i]
        eModel.buildKernelsMatrices()
        eLowerBoundValues[i] = eModel.eval()
    tParams[0][1] = tParamValue
    eParams[0][1] = eParamValue
    sTLowerBoundValues = np.sort(tLowerBoundValues)
    tYMin = sTLowerBoundValues[round(.1*len(tLowerBoundValues))]
    tYMax = sTLowerBoundValues[-1]
    # tYMin = -67000
    # tYMax = -58000
    eYMin = 650
    eYMax = 750
    # eYMin = 50
    # eYMax = 150
    fig = getPlotLowerBounds(paramValues=paramValues, tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel="Period Value", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
    fig.write_image(figFilenamePattern.format("Period", "png"))
    fig.write_html(figFilenamePattern.format("Period", "html"))
    pio.renderers.default = "browser"
    fig.show()

#     tEmbeddingParams = tModel.getSVEmbeddingParams()
#     tParamValue = tEmbeddingParams[0][0,0].item()
#     eEmbeddingParams = eModel.getSVEmbeddingParams()
#     eParamValue = eEmbeddingParams[0][0,0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     tYMin = -10200
#     tYMax = 0.0
#     eYMin = -500
#     eYMax = 500
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     tLowerBoundValues = np.empty(paramValues.shape)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tEmbeddingParams[0][0,0] = paramValues[i]
#         tLowerBoundValues[i] = tModel.eval()
#         eEmbeddingParams[0][0,0] = paramValues[i]
#         eLowerBoundValues[i] = eModel.eval()
#     tEmbeddingParams[0][0,0] = tParamValue
#     eEmbeddingParams[0][0,0] = eParamValue
#     fig = getPlotLowerBounds(paramValues=paramValues,
#                              tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel="C[0,0] Value", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
#     fig.write_image(figFilenamePattern.format("C00", "png"))
#     fig.write_html(figFilenamePattern.format("C00", "html"))
#     pio.renderers.default = "browser"
#     fig.show()

#     tEmbeddingParams = tModel.getSVEmbeddingParams()
#     tParamValue = tEmbeddingParams[1][0].item()
#     eEmbeddingParams = eModel.getSVEmbeddingParams()
#     eParamValue = eEmbeddingParams[1][0].item()
#     paramValueStart = -4
#     paramValueEnd = 4
#     paramValueStep = .01
#     tYMax = -2500
#     tYMin = -10200
#     eYMax = 300
#     eYMin = -700
#     paramValues = np.arange(paramValueStart, paramValueEnd, paramValueStep)
#     tLowerBoundValues = np.empty(paramValues.shape)
#     eLowerBoundValues = np.empty(paramValues.shape)
#     for i in range(len(paramValues)):
#         tEmbeddingParams[1][0] = paramValues[i]
#         tLowerBoundValues[i] = tModel.eval()
#         eEmbeddingParams[1][0] = paramValues[i]
#         eLowerBoundValues[i] = eModel.eval()
#     tEmbeddingParams[1][0] = tParamValue
#     eEmbeddingParams[1][0] = eParamValue
#     fig = getPlotLowerBounds(paramValues=paramValues, tLowerBoundValues=tLowerBoundValues, eLowerBoundValues=eLowerBoundValues, xlabel="d[0] Value", tParamValue=tParamValue, eParamValue=eParamValue, tYMin=tYMin, tYMax=tYMax, eYMin=eYMin, eYMax=eYMax,)
#     fig.write_image(figFilenamePattern.format("d0", "png"))
#     fig.write_html(figFilenamePattern.format("d0", "html"))
#     pio.renderers.default = "browser"
#     fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)

