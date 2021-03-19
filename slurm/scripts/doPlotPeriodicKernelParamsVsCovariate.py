
import sys
import pdb
import math
import numbers
import pickle
import pandas as pd
import numpy as np
import scipy.io
import argparse
import plotly.graph_objs as go
sys.path.append("../../src")

def getPythonPeriodicKernelParamsAtConvergence(modelFilenames, paramScale, paramToPlot="Period"):
    paramsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            with open(modelFilename, "rb") as f:
                loadRes = pickle.load(f)
            if paramToPlot=="Period":
                param = loadRes["model"].getKernelsParams()[0][1]/paramScale
            elif paramToPlot=="Lengthscale":
                param = loadRes["model"].getKernelsParams()[0][0]/paramScale
            else:
                raise ValueError("Invalid paramToPlot={:s}".format(paramToPlot))
            paramsAtConvergence[i] = param
        else:
            paramsAtConvergence[i] = float("nan")
    return paramsAtConvergence

def getMatlabPeriodicKernelParamsAtConvergence(modelFilenames, paramToPlot="Period"):
    paramsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            loadRes = scipy.io.loadmat(modelFilename)
            if paramToPlot=="Period":
                param = loadRes["m"][0,0]["kerns"][0,0]["hprs"][0,0].squeeze()[1]
            elif paramToPlot=="Lengthscale":
                param = loadRes["m"][0,0]["kerns"][0,0]["hprs"][0,0].squeeze()[0]
            else:
                raise ValueError("Invalid paramToPlot={:s}".format(paramToPlot))
            paramsAtConvergence[i] = param
        else:
            paramsAtConvergence[i] = float("nan")
    return paramsAtConvergence

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--periodicKernelParamToPlot", help="Periodic kernel parameter to plot (Period or Lengthscale)", default="Period")
    parser.add_argument("--generativeParam", help="Value of of the generative parameter", type=float, default=5.0)
    parser.add_argument("--pythonDescriptorsFilename", help="Name of file containing the descriptor of each line in the Python data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--matlabDescriptorsFilename", help="Name of file containing the descriptor of each line in the Matlab data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilenames", help="Filenames containing the labels and model estimation numbers of models to plot", default="../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv,../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv,../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv")
    parser.add_argument("--modelSetsTypes", help="Legend labels for each set of estimated models", default="Python,Python,Python,Matlab")
    parser.add_argument("--modelSetsLegendLabels", help="Legend labels for each set of estimated models", default="PyTorch LBFGS,SciPy L-BFGS-B,SciPy trust-ncg,Matlab minFunc")
    parser.add_argument("--paramScales", help="Scale for estimated parameters", default="1e3")
    parser.add_argument("--pythonModelFilenamePattern", help="Filename pattern of a Python models", default="../../scripts/{:s}")
    parser.add_argument("--matlabModelFilenamePattern", help="Filename pattern of a Matlab models", default="../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    parser.add_argument("--generativeParamLineDash", help="Line dash for generative period", default="dash")
    parser.add_argument("--xlab", help="Figure label for the abcissa", default="Period Parameter Initial Condition")
    parser.add_argument("--ylab", help="Figure label for the ordinate", default="Estimated Period")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/20secSim_estimatedPeriodVsPeriod0_multipleOptimMethods.{:s}")
    args = parser.parse_args()

    periodicKernelParamToPlot = args.periodicKernelParamToPlot
    generativeParam = args.generativeParam
    pythonDescriptorsFilename = args.pythonDescriptorsFilename
    matlabDescriptorsFilename = args.matlabDescriptorsFilename
    labelsAndEstNumbersFilenames = args.labelsAndEstNumbersFilenames
    modelSetsTypes = args.modelSetsTypes
    modelSetsLegendLabels = args.modelSetsLegendLabels
    paramScales = args.paramScales
    pythonModelFilenamePattern = args.pythonModelFilenamePattern
    matlabModelFilenamePattern = args.matlabModelFilenamePattern
    generativeParamLineDash = args.generativeParamLineDash
    xlab = args.xlab
    ylab = args.ylab
    figFilenamePattern = args.figFilenamePattern

    paramScalesSplitted = [float(aParamString) for aParamString in paramScales.split(",")]
    labelsAndEstNumbersFilenamesSplitted = labelsAndEstNumbersFilenames.split(",")
    modelSetsTypesSplitted = modelSetsTypes.split(",")
    modelSetsLegendLabelsSplitted = modelSetsLegendLabels.split(",")
    valuesList = [None]*len(labelsAndEstNumbersFilenamesSplitted)
    for i, labelsAndEstNumbersFilename in enumerate(labelsAndEstNumbersFilenamesSplitted):
        partialModelsFilenames = pd.read_csv(labelsAndEstNumbersFilename, sep=" ", header=0).iloc[:,1].tolist()
        fullModelsFilenames = []
        if modelSetsTypesSplitted[i]=="Python":
            for partialModelFilename in partialModelsFilenames:
                if not isinstance(partialModelFilename, numbers.Number):
                    fullModelsFilenames.append(pythonModelFilenamePattern.format(partialModelFilename))
                else:
                    fullModelsFilenames.append(partialModelFilename)
        elif modelSetsTypesSplitted[i]=="Matlab":
            for partialModelFilename in partialModelsFilenames:
                if not math.isnan(partialModelFilename):
                    fullModelsFilenames.append(matlabModelFilenamePattern.format(int(partialModelFilename)))
                else:
                    fullModelsFilenames.append(partialModelFilename)
        else:
            raise ValueError("{:s} is not a valid model set type".format(modelSetsTypesSplitted[i]))
        if modelSetsTypesSplitted[i]=="Python":
            values = getPythonPeriodicKernelParamsAtConvergence(modelFilenames=fullModelsFilenames, paramScale=paramScalesSplitted[i], paramToPlot=periodicKernelParamToPlot)
        elif modelSetsTypesSplitted[i]=="Matlab":
            values = getMatlabPeriodicKernelParamsAtConvergence(modelFilenames=fullModelsFilenames, paramToPlot=periodicKernelParamToPlot)
        else:
            raise ValueError("{:s} is not a valid model set type".format(modelSetsTypesSplitted[i]))
        valuesList[i] = values

    pythonDescriptors = pd.read_csv(pythonDescriptorsFilename, sep=" ").iloc[:,0].tolist()
    matlabDescriptors = pd.read_csv(matlabDescriptorsFilename, sep=" ").iloc[:,0].tolist()

    traces = []
    for i, values in enumerate(valuesList):
        if modelSetsTypesSplitted[i]=="Python":
            trace = go.Scatter(
                x = pythonDescriptors,
                y = values,
                mode="lines+markers",
                name=modelSetsLegendLabelsSplitted[i],
                showlegend=True,
            )
            traces.append(trace)
        elif modelSetsTypesSplitted[i]=="Matlab":
            trace = go.Scatter(
                x = matlabDescriptors,
                y = values,
                mode="lines+markers",
                name=modelSetsLegendLabelsSplitted[i],
                showlegend=True,
            )
            traces.append(trace)
        else:
            raise ValueError("{:s} is not a valid model set type".format(modelSetsTypesSplitted[i]))

    fig = go.Figure()
    for trace in traces:
        fig.add_trace(trace)
    fig.add_hline(y=generativeParam, line_dash=generativeParamLineDash)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
