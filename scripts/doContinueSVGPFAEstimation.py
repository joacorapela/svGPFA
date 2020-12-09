import sys
import os
import pdb
import random
import numpy as np
import pickle
import argparse
import configparser
sys.path.append("../src")
import stats.svGPFA.svEM
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("initialEstResNumber", help="estimation result number of the model to add iterations", type=int)
    parser.add_argument("nIter", help="number of iterations to add", type=int)
    parser.add_argument("--savePartial", action="store_true", help="save partial model estimates")
    parser.add_argument("--estimatedModelFilenamePattern", default="results/{:08d}_estimatedModel.pickle", help="estimated model filename pattern")
    parser.add_argument("--estimatedModelMetaDataFilenamePattern", default="results/{:08d}_estimation_metaData.ini", help="estimated model meta data filename pattern")
    parser.add_argument("--estimatedPartialModelFilenamePattern", default="results/{:08d}_{{:s}}_estimatedModel.pickle", help="estimated partial model filename pattern")
    args = parser.parse_args()

    initialEstResNumber = args.initialEstResNumber
    nIter = args.nIter
    savePartial = args.savePartial
    estimatedModelFilenamePattern = args.estimatedModelFilenamePattern
    estimatedModelMetaDataFilenamePattern = args.estimatedModelMetaDataFilenamePattern
    estimatedPartialModelFilenamePattern = args.estimatedPartialModelFilenamePattern

    initialEstModelMetaDataFilename = estimatedModelMetaDataFilenamePattern.format(initialEstResNumber)
    initialEstModelMetaDataConfig = configparser.ConfigParser()
    initialEstModelMetaDataConfig.read(initialEstModelMetaDataFilename)
    initialEstimationMetaDataFilename = initialEstModelMetaDataConfig["estimation_params"]["estimationMetaDataFilename".lower()]

    initialEstimationMetaDataConfig = configparser.ConfigParser()
    initialEstimationMetaDataConfig.read(initialEstimationMetaDataFilename)
    optimParamsDict = initialEstimationMetaDataConfig._sections["optim_params"]
    optimParams = utils.svGPFA.miscUtils.getOptimParams(optimParamsDict=optimParamsDict)

    initialModelFilename = estimatedModelFilenamePattern.format(initialEstResNumber)
    with open(initialModelFilename, "rb") as f: estResults = pickle.load(f)
    lowerBoundHist = estResults["lowerBoundHist"]
    elapsedTimeHist = estResults["elapsedTimeHist"]
    model = estResults["model"]

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM()
    lowerBoundHist, elapsedTimeHist = svEM.continueMaximization(model=model,
                                                                nIter=nIter,
                                                                lowerBoundHist=lowerBoundHist,
                                                                elapsedTimeHist=elapsedTimeHist,
                                                                optimParams=optimParams)

    estPrefixUsed = True
    while estPrefixUsed:
        finalEstResNumber = random.randint(0, 10**8)
        finalEstimResMetaDataFilename = estimatedModelMetaDataFilenamePattern.format(finalEstResNumber)
        if not os.path.exists(finalEstimResMetaDataFilename):
           estPrefixUsed = False
    finalModelSaveFilename = estimatedModelFilenamePattern.format(finalEstResNumber)

    # save estimated values
    finalEstimationMetaDataConfig = configparser.ConfigParser()
    finalEstimationMetaDataConfig["estimation_params"] = {"initialEstResNumber": initialEstResNumber, "nIter": nIter, "estimationMetaDataFilename": initialEstModelMetaDataFilename}
    with open(finalEstimResMetaDataFilename, "w") as f: finalEstimationMetaDataConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "model": model}
    with open(finalModelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
