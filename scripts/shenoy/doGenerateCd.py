import sys
import pdb
import argparse
import pandas as pd
import torch

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--nLatents", help="number of latents", default="[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]")
    parser.add_argument("--nNeurons", help="number of neurons", type=int, default=105)
    parser.add_argument("--Cmean", help="mean of the C matrix", type=float, default=0.0)
    parser.add_argument("--Cstd", help="std of the C matrix", type=float, default=0.01)
    parser.add_argument("--dMean", help="mean of the d vector", type=float, default=0.0)
    parser.add_argument("--dStd", help="std of the d vector", type=float, default=0.01)
    parser.add_argument("--Cfilename_pattern", help="filename patter for C matrix", default="data/C_Gaussian_means{:.02f}_stds{:.02f}_neurons{:03d}_latents{:02d}.csv")
    parser.add_argument("--dFilename_pattern", help="filename patter for d vector", default="data/d_Gaussian_means{:.02f}_stds{:.02f}_neurons{:03d}.csv")
    args = parser.parse_args()

    nLatents = [int(str) for str in args.nLatents[1:-1].split(",")]
    nNeurons = args.nNeurons
    Cmean = args.Cmean
    Cstd = args.Cstd
    dMean = args.dMean
    dStd = args.dStd
    Cfilename_pattern = args.Cfilename_pattern
    dFilename_pattern = args.dFilename_pattern

    for r in nLatents:
        C = torch.normal(mean=Cmean, std=Cstd, size=(nNeurons, r), dtype=torch.double)
        Cfilename = Cfilename_pattern.format(Cmean, Cstd, nNeurons, r)
        pd.DataFrame(C.numpy()).to_csv(Cfilename, header=False, index=False)
    d = torch.normal(mean=dMean, std=dStd, size=(nNeurons, 1), dtype=torch.double)
    dFilename = dFilename_pattern.format(dMean, dStd, nNeurons)
    pd.DataFrame(d.numpy()).to_csv(dFilename, header=False, index=False)
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
