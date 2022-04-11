
import sys
import pdb
import pickle
import argparse
sys.path.append("../src")

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estNumber", help="Estimation number", type=int)
    args = parser.parse_args()
    estNumber = args.estNumber
    modelSaveFilename = "results/{:08d}_estimatedModel.pickle".format(estNumber)
    with open(modelSaveFilename, "rb") as f: savedResults = pickle.load(f)
    model = savedResults["model"]
    lowerBound = model.eval()
    print("Achieved lower bound {:f}".format(lowerBound))

    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
