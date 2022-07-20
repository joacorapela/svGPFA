
import sys
import os
import torch
import pdb
import pickle
import argparse
import configparser
from scipy.io import loadmat
import plotly.io as pio
sys.path.append("../src")
import plot.svGPFA.plotUtilsPlotly
import utils.svGPFA.configUtils
import utils.svGPFA.initUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estNumber", help="Estimation number", type=int)
    args = parser.parse_args()
    pEstNumber = args.estNumber

    pEstimMetaDataFilename = "results/{:08d}_estimation_metaData.ini".format(pEstNumber)
    pEstConfig = configparser.ConfigParser()
    pEstConfig.read(pEstimMetaDataFilename)
    pSimNumber = int(pEstConfig["simulation_params"]["simResNumber"])

    pSimResMetaDataFilename = "results/{:08d}_simulation_metaData.ini".format(pSimNumber)
    pSimResMetaDataConfig = configparser.ConfigParser()
    pSimResMetaDataConfig.read(pSimResMetaDataFilename)
    pSimInitConfigFilename = pSimResMetaDataConfig["simulation_params"]["simInitConfigFilename"]

    pSimInitConfig = configparser.ConfigParser()
    pSimInitConfig.read(pSimInitConfigFilename)
    nLatents = int(pSimInitConfig["control_variables"]["nLatents"])
    tKernels = utils.svGPFA.configUtils.getKernels(nLatents=nLatents, config=pSimInitConfig, forceUnitScale=True)
    kernelsTypes = [type(tKernels[k]).__name__ for k in range(nLatents)]
    tKernelsParams = utils.svGPFA.initUtils.getKernelsParams0(kernels=tKernels, noiseSTD=0.0)

    pModelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(pEstNumber)
    figFilenamePattern = "figures/{:08d}_truePythonKernelsParamsPointProcess.{{:s}}".format(pEstNumber)

    with open(pModelSaveFilename, "rb") as f: res = pickle.load(f)
    pModel = res["model"]
    pKernelsParams = pModel.getKernelsParams()

    fig = plot.svGPFA.plotUtilsPlotly.getPlotTrueAndEstimatedKernelsParams(
        kernelsTypes=kernelsTypes,
        trueKernelsParams=tKernelsParams,
        estimatedKernelsParams=pKernelsParams)
    fig.write_image(figFilenamePattern.format("png"))
    fig.write_html(figFilenamePattern.format("html"))
    pio.renderers.default = "browser"
    fig.show()

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
