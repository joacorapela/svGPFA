
import sys
import pdb
import math
import numpy as np

def main(argv):
    nNeurons = 100
    nLatents = 3
    maxFiringRate = 40 # spikes/sec
    meanFiringRate = 20 # spikes/sec
    C_filename = "data/00000001_C_3Latents_100Neurons.csv"
    d_filename = "data/00000001_d_100Neurons.csv"

    d_amplitude = math.log(meanFiringRate)
    C_amplitude = math.log(maxFiringRate-meanFiringRate)
    C = np.zeros((nNeurons, nLatents))
    for i in range(nLatents-1):
        C[i*(nNeurons//nLatents)+np.arange(nNeurons//nLatents), i] = C_amplitude
    C[(nLatents-1)*(nNeurons//nLatents):, nLatents-1] = C_amplitude
    d = np.ones(nNeurons)*d_amplitude

    np.savetxt(C_filename, C, delimiter=",")
    np.savetxt(d_filename, d, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
