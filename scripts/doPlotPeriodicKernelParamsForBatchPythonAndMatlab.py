
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import scipy.io
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--pLabelsAndEstNumbersFilename", help="Filename containing the Python labels ane model estimation numbers of all models to plot", default="../slurm/data/labelsEstNumbers20secSim.csv")
    parser.add_argument("--mLabelsAndEstNumbersFilename", help="Filename containing the Matlab labels ane model estimation numbers of all models to plot", default="../../matlabCode/scripts/results/log20sec.csv")
    parser.add_argument("--pModelFilenamePattern", help="Filename of the pickle file where the Python model was saved", default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--mModelFilenamePattern", help="Filename of the mat file where a Matlab model was saved", default="../../matlabCode/scripts/results/{:08d}-pointProcessEstimationRes.mat")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="figures/periodicKernelParamsOfModelBatchPythonMatlab.{:s}")
    parser.add_argument("--xlabel", help="Figure xlabel", default="Period")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lengthscale")
    args = parser.parse_args()
    pLabelsAndEstNumbersFilename = args.pLabelsAndEstNumbersFilename
    mLabelsAndEstNumbersFilename = args.mLabelsAndEstNumbersFilename
    pModelFilenamePattern = args.pModelFilenamePattern
    mModelFilenamePattern = args.mModelFilenamePattern
    figFilenamePattern = args.figFilenamePattern
    xlabel = args.xlabel
    ylabel = args.ylabel

    fig = go.Figure()
    with open(pLabelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        for row in csvReader:
            label = row[0]
            estNumber = int(row[1])
            modelFilename = pModelFilenamePattern.format(estNumber)
            with open(modelFilename, "rb") as f: res = pickle.load(f)
            lowerBoundHistExt = [res["lowerBoundHist"][0]] + res["lowerBoundHist"]
            iterationModelParams = res["iterationModelParams"]
            trace = go.Scatter(
                x=iterationModelParams[:,1]/1000.0, # period
                y=iterationModelParams[:,0], # lengthscale
                # mode="lines+markers",
                mode="markers",
                name="{:s}_{:s}".format(label, "Python"),
                showlegend=True,
                hovertext=lowerBoundHistExt,
            )
            fig.add_trace(trace)
    with open(mLabelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        for row in csvReader:
            label = row[0]
            estNumber = int(row[1])
            modelFilename = mModelFilenamePattern.format(estNumber)
            res = scipy.io.loadmat(modelFilename)
            m_struct = res["m"]
            lowerBoundHist = (-1*m_struct["FreeEnergy"][0,0]).tolist()
            lowerBoundHistExt = [lowerBoundHist[0]] + lowerBoundHist
            # pdb.set_trace()
            iterationModelParams = m_struct["iterationsModelParams"][0,0]
            trace = go.Scatter(
                x=iterationModelParams[:,1], # period
                y=iterationModelParams[:,0], # lengthscale
                # mode="lines+markers",
                mode="markers",
                name="{:s}_{:s}".format(label, "Matlab"),
                showlegend=True,
                hovertext=lowerBoundHistExt,
            )
            fig.add_trace(trace)
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text=xlabel)

    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))

    fig.show()

if __name__=="__main__":
    main(sys.argv)
