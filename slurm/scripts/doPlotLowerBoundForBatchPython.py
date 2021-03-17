
import sys
import os
import pdb
import pickle
import argparse
import csv
import numpy as np
import plotly.graph_objs as go
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("figsPrefix", help="Figures prefix")
    parser.add_argument("labelsAndEstNumbersFilename", help="Filename containing the labels ane model estimation numbers of all models to plot")
    parser.add_argument("--modelFilenamePattern", help="Filename of the pickle file where the model was saved", default="results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--figFilenamePattern", help="Filename pattern of the plot figure", default="figures/{:s}_lowerBoundsVs{:s}OfModelBatch.{:s}")
    parser.add_argument("--xlabelIterationNumber", help="Iteration number figure xlabel", default="Iteration Number")
    parser.add_argument("--xlabelElapsedTime", help="Elapsed time igure xlabel", default="Elapsed Time (sec)")
    parser.add_argument("--ylabel", help="Figure ylabel", default="Lower Bound")
    args = parser.parse_args()
    figsPrefix = args.figsPrefix
    labelsAndEstNumbersFilename = args.labelsAndEstNumbersFilename
    modelFilenamePattern = args.modelFilenamePattern
    figFilenamePattern = args.figFilenamePattern
    xlabelIterationNumber = args.xlabelIterationNumber
    xlabelElapsedTime = args.xlabelElapsedTime
    ylabel = args.ylabel

    figIteration = go.Figure()
    figElapsedTime = go.Figure()
    with open(labelsAndEstNumbersFilename) as f:
        csvReader = csv.reader(f, delimiter=" ")
        for row in csvReader:
            label = row[0]
            estNumber = int(row[1])
            modelFilename = modelFilenamePattern.format(estNumber)
            with open(modelFilename, "rb") as f: res = pickle.load(f)
            lowerBoundHist = res["lowerBoundHist"]
            traceIteration = go.Scatter(
                x=np.arange(len(lowerBoundHist)),
                y=lowerBoundHist,
                mode="lines",
                name=label,
                showlegend=True,
            )
            figIteration.add_trace(traceIteration)
            elapsedTimeHist = res["elapsedTimeHist"]
            traceElapsedTime = go.Scatter(
                x=elapsedTimeHist,
                y=lowerBoundHist,
                mode="lines",
                name=label,
                showlegend=True,
            )
            figElapsedTime.add_trace(traceElapsedTime)
    figIteration.update_yaxes(title_text=ylabel)
    figIteration.update_xaxes(title_text=xlabelIterationNumber)
    figElapsedTime.update_yaxes(title_text=ylabel)
    figElapsedTime.update_xaxes(title_text=xlabelElapsedTime)

    pngIterationFigFilename = figFilenamePattern.format(figsPrefix, "Iteration", "png")
    htmlIterationFigFilename = figFilenamePattern.format(figsPrefix, "Iteration", "html")
    pngElapsedTimeFigFilename = figFilenamePattern.format(figsPrefix, "ElapsedTime", "png")
    htmlElapsedTimeFigFilename = figFilenamePattern.format(figsPrefix, "ElapsedTime", "html")
    if os.path.isfile(pngIterationFigFilename):
        raise ValueError("{:s} exists. Please provide aanother figure prefix".format(pngIterationFigFilename))
    if os.path.isfile(htmlIterationFigFilename):
        raise ValueError("{:s} exists. Please provide aanother figure prefix".format(htmlIterationFigFilename))
    if os.path.isfile(pngElapsedTimeFigFilename):
        raise ValueError("{:s} exists. Please provide aanother figure prefix".format(pngElapsedTimeFigFilename))
    if os.path.isfile(htmlElapsedTimeFigFilename):
        raise ValueError("{:s} exists. Please provide aanother figure prefix".format(htmlElapsedTimeFigFilename))

    figIteration.write_image(pngIterationFigFilename)
    figIteration.write_html(htmlIterationFigFilename)
    figElapsedTime.write_image(pngElapsedTimeFigFilename)
    figElapsedTime.write_html(htmlElapsedTimeFigFilename)

    figIteration.show()
    figElapsedTime.show()

if __name__=="__main__":
    main(sys.argv)
