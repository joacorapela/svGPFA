
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

def getPythonElapsedTimesAtConvergence(modelFilenames, convergenceTolerance):
    elapsedTimesAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            with open(modelFilename, "rb") as f:
                loadRes = pickle.load(f)
            lowerBoundHist = np.array(loadRes["lowerBoundHist"])
            elapsedTimeHist = np.array(loadRes["elapsedTimeHist"])
            lowerBoundsDiff = lowerBoundHist[1:]-lowerBoundHist[:-1]
            indices = np.where(lowerBoundsDiff<convergenceTolerance)[0]
            if len(indices)>0:
                elapsedTimesAtConvergence[i] = elapsedTimeHist[indices[0]]
            else:
                elapsedTimesAtConvergence[i] = elapsedTimeHist[-1]
        else:
                elapsedTimesAtConvergence[i] = float("nan")
    return elapsedTimesAtConvergence

def getMatlabElapsedTimesAtConvergence(modelFilenames, convergenceTolerance):
    elapsedTimesAtConvergence = [None]*len(modelFilenames)
    for i, modelFilename in enumerate(modelFilenames):
        if not isinstance(modelFilename, numbers.Number):
            loadRes = scipy.io.loadmat(modelFilename)
            lowerBoundHist = loadRes["lowerBound"][:,0]
            elapsedTimeHist = loadRes["elapsedTime"][:,0]
            lowerBoundsDiff = lowerBoundHist[1:]-lowerBoundHist[:-1]
            indices = np.where(lowerBoundsDiff<convergenceTolerance)[0]
            if len(indices)>0:
                elapsedTimesAtConvergence[i] = elapsedTimeHist[indices[0]]
            else:
                elapsedTimesAtConvergence[i] = elapsedTimeHist[-1]
        else:
                elapsedTimesAtConvergence[i] = float("nan")
    return elapsedTimesAtConvergence

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythonDescriptorsFilename", help="Name of file containing the descriptor of each line in the Python data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--matlabDescriptorsFilename", help="Name of file containing the descriptor of each line in the Matlab data files", default="../../../matlab/slurm/data/descriptors_20secSim.csv")
    parser.add_argument("--labelsAndEstNumbersFilename_PyTorch_LBFGS", help="Filename containing the labels and model estimation numbers of the PyTorch LBFGS models to plot", default="../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv")
    parser.add_argument("--labelsAndEstNumbersFilename_SciPy_L-BFGS-B", help="Filename containing the labels and model estimation numbers of the SciPy_L-BFGS-B models to plot", default="../data/labelsEstNumbers_20secSim_SciPy_L-BFGS-B.csv")
    parser.add_argument("--labelsAndEstNumbersFilename_SciPy_trust-ncg", help="Filename containing the labels and model estimation numbers of the Scipy_trust-ncg models to plot", default="../data/labelsEstNumbers_20secSim_SciPy_trust-ncg.csv")
    parser.add_argument("--labelsAndEstNumbersFilename_Matlab_minFunc", help="Filename containing the labels and model estimation numbers of the Matlab minFunc models to plot", default="../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv")
    parser.add_argument("--pythonModelFilenamePattern", help="Filename pattern of a Python models", default="../../scripts/{:s}")
    parser.add_argument("--matlabModelFilenamePattern", help="Filename pattern of a Matlab models", default="../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    parser.add_argument("--convergenceTolerance", help="Lower boundd convergence tolerance", type=float, default=1e-5)
    parser.add_argument("--xlab", help="Figure label for the abcissa", default="Period Parameter Initial Condition")
    parser.add_argument("--ylab", help="Figure label for the ordinate", default="Elapsed Time (sec)")
    parser.add_argument("--figFilenamePattern", help="Figure filename pattern", default="../figures/20secSim_elapsedTimes.{:s}")
    args = parser.parse_args()

    pythonDescriptorsFilename = args.pythonDescriptorsFilename
    matlabDescriptorsFilename = args.matlabDescriptorsFilename
    labelsAndEstNumbersFilename_PyTorch_LBFGS = args.labelsAndEstNumbersFilename_PyTorch_LBFGS
    labelsAndEstNumbersFilename_SciPy_L_BFGS_B = args.labelsAndEstNumbersFilename_SciPy_L_BFGS_B
    labelsAndEstNumbersFilename_SciPy_trust_ncg = args.labelsAndEstNumbersFilename_SciPy_trust_ncg
    labelsAndEstNumbersFilename_Matlab_minFunc = args.labelsAndEstNumbersFilename_Matlab_minFunc
    pythonModelFilenamePattern = args.pythonModelFilenamePattern
    matlabModelFilenamePattern = args.matlabModelFilenamePattern
    convergenceTolerance = args.convergenceTolerance
    xlab = args.xlab
    ylab = args.ylab
    figFilenamePattern = args.figFilenamePattern

    partialModelsFilenames = pd.read_csv(labelsAndEstNumbersFilename_PyTorch_LBFGS, sep=" ").iloc[:,1].tolist()
    fullModelsFilenames = []
    for partialModelFilename in partialModelsFilenames:
        if not isinstance(partialModelFilename, numbers.Number):
            fullModelsFilenames.append(pythonModelFilenamePattern.format(partialModelFilename))
        else:
            fullModelsFilenames.append(partialModelFilename)
    PyTorch_LBFGS_values = getPythonElapsedTimesAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)

    partialModelsFilenames = pd.read_csv(labelsAndEstNumbersFilename_SciPy_L_BFGS_B, sep=" ").iloc[:,1].tolist()
    fullModelsFilenames = []
    for partialModelFilename in partialModelsFilenames:
        if not isinstance(partialModelFilename, numbers.Number):
            fullModelsFilenames.append(pythonModelFilenamePattern.format(partialModelFilename))
        else:
            fullModelsFilenames.append(partialModelFilename)
    SciPy_L_BFGS_B_values = getPythonElapsedTimesAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)

    partialModelsFilenames = pd.read_csv(labelsAndEstNumbersFilename_SciPy_trust_ncg, sep=" ").iloc[:,1].tolist()
    fullModelsFilenames = []
    for partialModelFilename in partialModelsFilenames:
        if not isinstance(partialModelFilename, numbers.Number):
            fullModelsFilenames.append(pythonModelFilenamePattern.format(partialModelFilename))
        else:
            fullModelsFilenames.append(partialModelFilename)
    SciPy_trust_ncg_values = getPythonElapsedTimesAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)

    estNumbers = pd.read_csv(labelsAndEstNumbersFilename_Matlab_minFunc, sep=" ").iloc[:,1].tolist()
    fullModelsFilenames = []
    for estNumber in estNumbers:
        if not math.isnan(estNumber):
            fullModelsFilenames.append(matlabModelFilenamePattern.format(estNumber))
        else:
            fullModelsFilenames.append(estNumber)
    Matlab_minFunc_values = getMatlabElapsedTimesAtConvergence(modelFilenames=fullModelsFilenames, convergenceTolerance=convergenceTolerance)

    pythonDescriptors = pd.read_csv(pythonDescriptorsFilename, sep=" ").iloc[:,0].tolist()
    matlabDescriptors = pd.read_csv(matlabDescriptorsFilename, sep=" ").iloc[:,0].tolist()


    trace_PyTorch_LBFGS = go.Scatter(
        x = pythonDescriptors,
        y = PyTorch_LBFGS_values,
        mode="lines+markers",
        name="Pytorch LBFGS",
        showlegend=True,
    )
    trace_SciPy_L_BFGS_B = go.Scatter(
        x = pythonDescriptors,
        y = SciPy_L_BFGS_B_values,
        mode="lines+markers",
        name="SciPy L-BFGS-B",
        showlegend=True,
    )
    trace_SciPy_trust_ncg = go.Scatter(
        x = pythonDescriptors,
        y = SciPy_trust_ncg_values,
        mode="lines+markers",
        name="SciPy trust-ncg",
        showlegend=True,
    )
    trace_Matlab_minFunc = go.Scatter(
        x = matlabDescriptors,
        y = Matlab_minFunc_values,
        mode="lines+markers",
        name="Matlab minFunc",
        showlegend=True,
    )

    fig = go.Figure()
    fig.add_trace(trace_PyTorch_LBFGS)
    fig.add_trace(trace_SciPy_L_BFGS_B)
    fig.add_trace(trace_SciPy_trust_ncg)
    fig.add_trace(trace_Matlab_minFunc)
    fig.update_xaxes(title_text=xlab)
    fig.update_yaxes(title_text=ylab)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
