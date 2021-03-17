
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import scipy.io
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
sys.path.append("../../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pythonDescriptorsFilename", help="Name of file containing the descriptor of each line in the Python data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--matlabDescriptorsFilename", help="Name of file containing the descriptor of each line in the Matlab data files", default="../data/descriptors_20secSim.csv")
    parser.add_argument("--pLabelsAndEstNumbersFilename", help="Filename containing the Python labels ane model estimation numbers of all models to plot", default="../data/labelsEstNumbers_20secSim_PyTorch_LBFGS.csv")
    parser.add_argument("--mLabelsAndEstNumbersFilename", help="Filename containing the Matlab labels ane model estimation numbers of all models to plot", default="../../../matlabCode/slurm/data/labelsEstNumbers_20secSim.csv")
    parser.add_argument("--pModelFilenamePattern", help="Filename of the pickle file where the Python model was saved", default="../../scripts/{:s}")
    parser.add_argument("--mModelFilenamePattern", help="Filename of the mat file where a Matlab model was saved", default="../../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    parser.add_argument("--generativePeriod", help="Generative period value", type=float, default=5.0)
    parser.add_argument("--generativeLengthscale", help="Generative lengthscale value", type=float, default=2.25)
    parser.add_argument("--generativeParamsLineDash", help="Line dash for generative params", default="dash")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="../figures/periodicKernelParamsOfModelBatchPythonMatlab.{:s}")
    parser.add_argument("--xlabel", help="Figure xlabel", default="Period")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lengthscale")

    args = parser.parse_args()
    pythonDescriptorsFilename = args.pythonDescriptorsFilename
    matlabDescriptorsFilename = args.matlabDescriptorsFilename
    pLabelsAndEstNumbersFilename = args.pLabelsAndEstNumbersFilename
    mLabelsAndEstNumbersFilename = args.mLabelsAndEstNumbersFilename
    pModelFilenamePattern = args.pModelFilenamePattern
    mModelFilenamePattern = args.mModelFilenamePattern
    generativePeriod = float(args.generativePeriod)
    generativeLengthscale = float(args.generativeLengthscale)
    generativeParamsLineDash = args.generativeParamsLineDash
    figFilenamePattern = args.figFilenamePattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    pythonDescriptors = pd.read_csv(pythonDescriptorsFilename, sep=" ", header=None).iloc[:,0].tolist()
    matlabDescriptors = pd.read_csv(matlabDescriptorsFilename, sep=" ", header=None).iloc[:,0].tolist()
    colorsList = plotly.colors.qualitative.Plotly

    fig = go.Figure()
    with open(pLabelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        i = 0
        for row in csvReader:
            label = pythonDescriptors[i]
            partialModelFilename = row[1]
            if len(partialModelFilename)>0:
                modelFilename = pModelFilenamePattern.format(partialModelFilename)
                with open(modelFilename, "rb") as f: res = pickle.load(f)
                lowerBoundHistExt = [res["lowerBoundHist"][0]] + res["lowerBoundHist"]
                iterationModelParams = res["iterationModelParams"]
                color = colorsList[i%len(colorsList)]
                trace = go.Scatter(
                    x=iterationModelParams[:,1]/1000.0, # period
                    y=iterationModelParams[:,0], # lengthscale
                    # mode="lines+markers",
                    mode="markers",
                    marker={"color": color, "symbol": "square"},
                    name="{:.1f}_{:s}".format(label, "Python"),
                    showlegend=True,
                    hovertext=lowerBoundHistExt,
                )
                # trace.update(marker_symbol="square")
                fig.add_trace(trace)
            i += 1
    with open(mLabelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        i = 0
        for row in csvReader:
            if len(row)==2:
                estNumber = int(row[1])
                modelFilename = mModelFilenamePattern.format(estNumber)
                label = matlabDescriptors[i]
                res = scipy.io.loadmat(modelFilename)
                m_struct = res["m"]
                lowerBoundHist = (-1*m_struct["FreeEnergy"][0,0]).tolist()
                lowerBoundHistExt = [lowerBoundHist[0]] + lowerBoundHist
                iterationModelParams = m_struct["iterationsModelParams"][0,0]
                color = colorsList[i%len(colorsList)]
                trace = go.Scatter(
                    x=iterationModelParams[:,1], # period
                    y=iterationModelParams[:,0], # lengthscale
                    # mode="lines+markers",
                    mode="markers",
                    marker={"color": color, "symbol": "circle-open"},
                    name="{:.1f}_{:s}".format(label, "Matlab"),
                    showlegend=True,
                    hovertext=lowerBoundHistExt,
                )
                # trace.update(marker_symbol="circle-open")
                fig.add_trace(trace)
            i += 1
    fig.add_vline(x=generativePeriod, line_dash=generativeParamsLineDash)
    fig.add_hline(y=generativeLengthscale, line_dash=generativeParamsLineDash)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
