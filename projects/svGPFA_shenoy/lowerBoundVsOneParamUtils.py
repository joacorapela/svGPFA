
def getReferenceParam(paramType, model, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
    if paramType=="kernel":
        kernelsParams = model.getKernelsParams()
        refParam = kernelsParams[latent][kernelParamIndex]
    elif paramType=="embeddingC":
        embeddingParams = model.getSVEmbeddingParams()
        refParam = embeddingParams[0][neuron,latent]
    elif paramType=="embeddingD":
        embeddingParams = model.getSVEmbeddingParams()
        refParam = embeddingParams[1][neuron]
    else:
        raise ValueError("Invalid paramType: {:s}".format(paramType))
    answer = refParam.clone()
    return answer

def getParamUpdateFun(paramType):
    def updateKernelParam(model, paramValue, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
        kernelsParams = model.getKernelsParams()
        kernelsParams[latent][kernelParamIndex] = paramValue
        # import pdb; pdb.set_trace()
        model.buildKernelsMatrices()

    def updateCParam(model, paramValue, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
        embeddingParams = model.getSVEmbeddingParams()
        embeddingParams[0][neuron,latent] = paramValue

    def updateDParam(model, paramValue, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
        embeddingParams = model.getSVEmbeddingParams()
        embeddingParams[1][neuron] = paramValue

    if paramType=="kernel":
        return updateKernelParam
    elif paramType=="embeddingC":
        return updateCParam
    elif paramType=="embeddingD":
        return updateDParam
    else:
        raise ValueError("Invalid paramType: {:s}".format(paramType))

def getKernelParamTypeString(kernelParamIndex):
        if kernelParamIndex==0:
            paramTypeString = "Lengthscale"
        elif kernelParamIndex==1:
            paramTypeString = "Period"
        else:
            raise ValueError("Invalid kernelParamIndex {:d}".format(kernelParamIndex))
        return paramTypeString

def getParamTitle(paramType, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2, indPointsLocsKMSRegEpsilon):
    if paramType=="kernel":
        kernelParamTypeString = getKernelParamTypeString(kernelParamIndex=kernelParamIndex)
        title = "Kernel {:s}, Latent {:d}, Epsilon {:f}".format(kernelParamTypeString, latent, indPointsLocsKMSRegEpsilon)
    elif paramType=="embeddingC":
        title = "Embedding Mixing Matrix, Neuron {:d}, Latent{:d}, Epsilon {:f}".format(neuron, latent), indPointsLocsKMSRegEpsilon
    elif paramType=="embeddingD":
        title = "Embedding offset vector, Neuron {:d}, Epsilon {:f}".format(neuron, indPointsLocsKMSRegEpsilon)
    else:
        raise ValueError("Invalid paramType: {:s}".format(paramType))
    return title

def getFigFilenamePattern(prefixNumber, descriptor, paramType, indPointsLocsKMSRegEpsilon, trial, latent, neuron, kernelParamIndex, indPointIndex, indPointIndex2):
    if paramType=="kernel":
        kernelParamTypeString = getKernelParamTypeString(kernelParamIndex=kernelParamIndex)
        figFilename = "figures/{:08d}_{:s}_epsilon{:f}_kernel_{:s}_latent{:d}.{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, kernelParamTypeString, latent)
    elif paramType=="embeddingC":
        figFilename = "figures/{:08d}_{:s}_epsilon{:f}_C[{:d},{:d}].{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, neuron, latent)
    elif paramType=="embeddingD":
        figFilename = "figures/{:08d}_{:s}_epsilon{:f}_d[{:d}].{{:s}}".format(prefixNumber, descriptor, indPointsLocsKMSRegEpsilon, neuron)
    else:
        raise ValueError("Invalid paramType: {:s}".format(paramType))
    return figFilename

