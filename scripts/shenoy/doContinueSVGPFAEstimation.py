import sys
import os
import pdb
import random
import numpy as np
import pickle
import argparse
import configparser
sys.path.append("../../src")
import stats.svGPFA.svEM
import utils.svGPFA.miscUtils

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("initialEstResNumber", help="estimation result number of the model to add iterations", type=int)
    parser.add_argument("nIter", help="number of iterations to add", type=int)
    parser.add_argument("--estimatedModelFilenamePattern", default="results/{:08d}_estimatedModel.pickle", help="estimated model filename pattern")
    parser.add_argument("--estimationMetaDataFilenamePattern", default="data/{:08d}_estimation_metaData.ini", help="estimation model meta data filename pattern")
    parser.add_argument("--estimatedModelMetaDataFilenamePattern", default="results/{:08d}_estimation_metaData.ini", help="estimated model meta data filename pattern")
    args = parser.parse_args()

    initialEstResNumber = args.initialEstResNumber
    nIter = args.nIter
    estimatedModelFilenamePattern = args.estimatedModelFilenamePattern
    estimationMetaDataFilenamePattern = args.estimationMetaDataFilenamePattern
    estimatedModelMetaDataFilenamePattern = args.estimatedModelMetaDataFilenamePattern

    initialEstModelMetaDataFilename = estimatedModelMetaDataFilenamePattern.format(initialEstResNumber)
    initialEstModelMetaDataConfig = configparser.ConfigParser()
    initialEstModelMetaDataConfig.read(initialEstModelMetaDataFilename)
    initialEstimationInitNumber = int(initialEstModelMetaDataConfig["estimation_params"]["estInitNumber".lower()])
    estMetaDataFilename = estimationMetaDataFilenamePattern.format(initialEstimationInitNumber)

    initialEstimationMetaDataConfig = configparser.ConfigParser()
    initialEstimationMetaDataConfig.read(estMetaDataFilename)
    optimParamsDict = initialEstimationMetaDataConfig._sections["optim_params"]
    optimParams = utils.svGPFA.miscUtils.getOptimParams(optimParamsDict=optimParamsDict)
    optimParams["em_max_iter"] = nIter

    initialModelFilename = estimatedModelFilenamePattern.format(initialEstResNumber)
    with open(initialModelFilename, "rb") as f: estResults = pickle.load(f)
    model = estResults["model"]

    estPrefixUsed = True
    while estPrefixUsed:
        finalEstResNumber = random.randint(0, 10**8)
        finalEstimResMetaDataFilename = estimatedModelMetaDataFilenamePattern.format(finalEstResNumber)
        if not os.path.exists(finalEstimResMetaDataFilename):
           estPrefixUsed = False
    finalModelSaveFilename = estimatedModelFilenamePattern.format(finalEstResNumber)

    # maximize lower bound
    svEM = stats.svGPFA.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optimParams=optimParams)

    # save estimated values
    finalEstimationMetaDataConfig = configparser.ConfigParser()
    finalEstimationMetaDataConfig["estimation_params"] = {"initialEstResNumber": initialEstResNumber, "nIter": nIter, "estInitNumber": initialEstimationInitNumber}
    with open(finalEstimResMetaDataFilename, "w") as f: finalEstimationMetaDataConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist, "elapsedTimeHist": elapsedTimeHist, "terminationInfo": terminationInfo, "iterationModelParams": iterationsModelParams, "model": model}
    with open(finalModelSaveFilename, "wb") as f: pickle.dump(resultsToSave, f)

    pdb.set_trace()

if __name__ == "__main__":
    main(sys.argv)
