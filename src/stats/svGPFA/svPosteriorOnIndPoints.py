
import pdb
import math
import torch
import utils.svGPFA.miscUtils

class SVPosteriorOnIndPoints:

    def __init__(self):
        super(SVPosteriorOnIndPoints, self).__init__()

    def setInitialParams(self, initialParams):
        nLatents = len(initialParams["qMu0"])
        self._qMu = [initialParams["qMu0"][k] for k in range(nLatents)]
        self._srQSigmaVecs = [initialParams["srQSigma0Vecs"][k] for k in range(nLatents)]

    def getParams(self):
        listOfTensors = []
        listOfTensors.extend([self._qMu[k] for k in range(len(self._qMu))])
        listOfTensors.extend([self._srQSigmaVecs[k] for k in range(len(self._srQSigmaVecs))])
        return listOfTensors

    def getQMu(self):
        return self._qMu

    def buildQSigma(self):
        # begin patch for older version of the code
#         if hasattr(self, "_qSRSigmaVec"):
#             self._srQSigmaVecs = [self._qSRSigmaVec[k].unsqueeze(-1) for k in range(len(self._qSRSigmaVec))]
#         elif self._srQSigmaVecs[0].dim()==2:
#             self._srQSigmaVecs = [self._srQSigmaVecs[k].unsqueeze(-1) for k in range(len(self._srQSigmaVecs))]
        # end patch for older version of the code
        qSigma = utils.svGPFA.miscUtils.buildQSigmasFromSRQSigmaVecs(srQSigmaVecs=self._srQSigmaVecs)
        return qSigma

    def get_flattened_params(self):
        flattened_params = torch.tensor([], dtype=torch.double)
        for k in range(len(self._qMu)):
            flattened_params = torch.cat((flattened_params, self._qMu[k].flatten()))
        for k in range(len(self._srQSigmaVecs)):
            flattened_params = torch.cat((flattened_params, self._srQSigmaVecs[k].flatten()))
        return flattened_params

    def get_flattened_params_grad(self):
        flattened_params_grad = torch.tensor([], dtype=torch.double)
        for k in range(len(self._qMu)):
            flattened_params_grad = torch.cat((flattened_params_grad, self._qMu[k].grad.flatten()))
        for k in range(len(self._srQSigmaVecs)):
            flattened_params_grad = torch.cat((flattened_params_grad, self._srQSigmaVecs[k].grad.flatten()))
        return flattened_params_grad

    def set_params_from_flattened(self, flattened_params):
        for k in range(len(self._qMu)):
            flattened_param = flattened_params[:self._qMu[k].numel()]
            self._qMu[k] = flattened_param.reshape(self._qMu[k].shape)
            flattened_params = flattened_params[self._qMu[k].numel():]
        for k in range(len(self._srQSigmaVecs)):
            flattened_param = flattened_params[:self._srQSigmaVecs[k].numel()]
            self._srQSigmaVecs[k] = flattened_param.reshape(self._srQSigmaVecs[k].shape)
            flattened_params = flattened_params[self._srQSigmaVecs[k].numel():]

    def set_params_requires_grad(self, requires_grad):
        for k in range(len(self._qMu)):
            self._qMu[k].requires_grad = requires_grad
        for k in range(len(self._srQSigmaVecs)):
            self._srQSigmaVecs[k].requires_grad = requires_grad

