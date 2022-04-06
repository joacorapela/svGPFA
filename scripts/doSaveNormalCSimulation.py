
import sys
import pdb
import argparse
import math
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("nNeurons", help="Number of neurons", type=int)
    parser.add_argument("nLatents", help="Number of latents", type=int)
    parser.add_argument("--mean", help="Mean of C values", type=float, default=0.0)
    parser.add_argument("--std", help="Standard deviation of C values", type=float, default=0.2)
    args = parser.parse_args()

    nNeurons = args.nNeurons
    nLatents = args.nLatents
    mean = args.mean
    std = args.std

    C_filename = "data/C_normal_{:.2f}mean_{:.02f}std_{:03d}neurons_{:02}latents.csv".format(mean, std, nNeurons, nLatents)

    C = np.random.normal(loc=mean, scale=std, size=(nNeurons, nLatents))

    np.savetxt(C_filename, C, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
