
import sys
import pdb
import math
import numpy as np

def main(argv):
    nNeurons = 20
    nLatents = 2
    dt = 1e-3
    maxFiringRate = 80000 # spikes/sec
    meanFiringRate = 20000 # spikes/sec
    C_filename = "data/00000000_C_2Latents_20Neurons.csv"
    d_filename = "data/00000000_d_20Neurons.csv"

    d_amplitude = math.log(meanFiringRate*dt)
    C_amplitude = math.log(maxFiringRate/meanFiringRate)
    C = np.random.uniform(low=0.0, high=1.0, size=(nNeurons, nLatents))
    rowSums = np.expand_dims(C.sum(axis=1), 1)
    C = C/rowSums
    C = C*C_amplitude
    d = np.ones(nNeurons)*d_amplitude
#     for i in range(nLatents-1):
#         C[i*(nNeurons//nLatents)+np.arange(nNeurons//nLatents), i] = C_amplitude
#     C[(nLatents-1)*(nNeurons//nLatents):, nLatents-1] = C_amplitude
#     d = np.ones(nNeurons)*d_amplitude

    np.savetxt(C_filename, C, delimiter=",")
    np.savetxt(d_filename, d, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
