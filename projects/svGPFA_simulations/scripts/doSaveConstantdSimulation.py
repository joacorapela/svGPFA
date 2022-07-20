
import sys
import pdb
import argparse
import math
import numpy as np

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("constant", help="Constant value", type=float)
    parser.add_argument("nNeurons", help="Number of neurons", type=int)
    args = parser.parse_args()

    constant = args.constant
    nNeurons = args.nNeurons

    d_filename = "data/d_constant_{:.2f}constant_{:03d}neurons.csv".format(constant, nNeurons)

    d = np.ones(nNeurons)*constant

    np.savetxt(d_filename, d, delimiter=",")

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
