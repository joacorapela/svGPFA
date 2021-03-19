
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

def getPythonLowerBoundsAtConvergence(modelFilenames, convergenceTolerance):
    lowerBoundsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            with open(modelFilename, "rb") as f:
                loadRes = pickle.load(f)
            lowerBoundHist = np.array(loadRes["lowerBoundHist"])
            lowerBoundsDiff = lowerBoundHist[1:]-lowerBoundHist[:-1]
            indices = np.where(lowerBoundsDiff<convergenceTolerance)[0]
            if len(indices)>0:
                lowerBoundsAtConvergence[i] = lowerBoundHist[indices[0]]
            else:
                lowerBoundsAtConvergence[i] = lowerBoundHist[-1]
        else:
                lowerBoundsAtConvergence[i] = float("nan")
    return lowerBoundsAtConvergence

def getMatlabLowerBoundsAtConvergence(modelFilenames, convergenceTolerance):
    lowerBoundsAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            loadRes = scipy.io.loadmat(modelFilename)
            lowerBoundHist = loadRes["lowerBound"][:,0]
            lowerBoundsDiff = lowerBoundHist[1:]-lowerBoundHist[:-1]
            indices = np.where(lowerBoundsDiff<convergenceTolerance)[0]
            if len(indices)>0:
                lowerBoundsAtConvergence[i] = lowerBoundHist[indices[0]]
            else:
                lowerBoundsAtConvergence[i] = lowerBoundHist[-1]
        else:
                lowerBoundsAtConvergence[i] = float("nan")
    return lowerBoundsAtConvergence

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythonDescriptorsFilename", help="Name of file containing the descriptor of each line in the Python data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--matlabDescriptorsFilename", help="Name of file containing the descriptor of each line in the Matlab data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilenames", help="Filenames containing the labels and model estimation numbers of models to plot", default="../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv,../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv,../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv,../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B_MAP.csv")
    parser.add_argument("--modelSetsTypes", help="Legend labels for each set of estimated models", default="Python,Python,Python,Matlab,Python")
    parser.add_argument("--modelSetsLegendLabels", help="Legend labels for each set of estimated models", default="PyTorch LBFGS,SciPy L-BFGS-B,SciPy trust-ncg,Matlab LBFGS,SciPy L-BFGS-B MAP")
    parser.add_argument("--pythonModelFilenamePattern", help="Filename pattern of a Python models", default="../../scripts/{:s}")
    parser.add_argument("--matlabModelFilenamePattern", help="Filename pattern of a Matlab models", default="../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    parser.add_argument("--convergenceTolerance", help="Lower boundd convergence tolerance", type=float, default=1e-5)
    parser.add_argument("--xlab", help="Figure label for the abcissa", default="Period Parameter Initial Condition")
    parser.add_argument("--ylab", help="Figure label for the ordinate", default="Lower Bound")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/20secSim_lowerBoundsVsPeriod0_multipleMethods.{:s}")
    args = parser.parse_args()

    pythonDescriptorsFilename = args.pythonDescriptorsFilename
    matlabDescriptorsFilename = args.matlabDescriptorsFilename
    labelsAndEstNumbersFilenames = args.labelsAndEstNumbersFilenames
    modelSetsTypes = args.modelSetsTypes
    modelSetsLegendLabels = args.modelSetsLegendLabels
    pythonModelFilenamePattern = args.pythonModelFilenamePattern
    matlabModelFilenamePattern = args.matlabModelFilenamePattern
    convergenceTolerance = args.convergenceTolerance
    xlab = args.xlab
    ylab = args.ylab
    figFilenamePattern = args.figFilenamePattern

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
            values = getPythonLowerBoundsAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)
        elif modelSetsTypesSplitted[i]=="Matlab":
            values = getMatlabLowerBoundsAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)
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
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

if __name__=="__main__":
    main(sys.argv)
