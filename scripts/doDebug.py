import pdb
import sys
import pickle
sys.path.append("../src")

def main(argv):
    with open("results/05393918_estimatedModel.pickle", "rb") as f:
        maxRes = pickle.load(f)
    model = maxRes["model"]
    pdb.set_trace()
    model.eval()
    pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
