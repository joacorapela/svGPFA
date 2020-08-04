
import sys
import pdb
import argparse
import math
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("nNeurons", help="Number of neurons", type=int)
    parser.add_argument("nLatents", help="Number of latents", type=int)
    parser.add_argument("maxCIF", help="Maximum of CIF", type=float)
    parser.add_argument("meanCIF", help="Mean of CIF", type=float)
    args = parser.parse_args()

    nNeurons = args.nNeurons
    nLatents = args.nLatents
    maxCIF = args.maxCIF
    meanCIF = args.meanCIF

    C_filename = "data/C_{:03d}neurons_{:02}latents_{:.02f}maxCIF_{:.02f}meanCIF.csv".format(nNeurons, nLatents, maxCIF, meanCIF)
    d_filename = "data/d_{:03d}neurons_{:02}latents_{:.02f}maxCIF_{:.02f}meanCIF.csv".format(nNeurons, nLatents, maxCIF, meanCIF)

    d_amplitude = math.log(meanCIF)
    C_maxAmplitude = (math.log(maxCIF)-math.log(meanCIF))/nLatents
    C = np.random.uniform(low=-1.0, high=1.0, size=(nNeurons, nLatents))*C_maxAmplitude
    d = np.ones(nNeurons)*d_amplitude

    np.savetxt(C_filename, C, delimiter=",")
    np.savetxt(d_filename, d, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
