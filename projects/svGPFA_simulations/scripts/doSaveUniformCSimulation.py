
import sys
import pdb
import argparse
import math
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("nNeurons", help="Number of neurons", type=int)
    parser.add_argument("nLatents", help="Number of latents", type=int)
    args = parser.parse_args()

    nNeurons = args.nNeurons
    nLatents = args.nLatents

    C_filename = "data/C_uniform_{:03d}neurons_{:02}latents.csv".format(nNeurons, nLatents)

    C = np.random.uniform(low=-1.0, high=1.0, size=(nNeurons, nLatents))

    np.savetxt(C_filename, C, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
