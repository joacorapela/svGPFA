import lowerBoundVsOneParamUtils

def getFigFilenamePattern(prefixNumber, descriptor, paramType, indPointsLocsKMSRegEpsilon, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
    if paramType=="kernel":
        kernelParamTypeString = lowerBoundVsOneParamUtils.getKernelParamTypeString(kernelParamIndex=kernelParamIndex)
        figFilename = "figures/{:08d}_condNumber_{:s}_epsilon{:f}_kernel_{:s}_latent{:d}.{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, kernelParamTypeString, latent)
    elif paramType=="embeddingC":
        figFilename = "figures/{:08d}_{:s}_epsilon{:f}_C[{:d},{:d}].{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, neuron, latent)
    elif paramType=="embeddingD":
        figFilename = "figures/{:08d}_{:s}_epsilon{:f}_d[{:d}].{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, neuron)
    else:
        raise ValueError("Invalid paramType: {:s}".format(paramType))
    return figFilename

