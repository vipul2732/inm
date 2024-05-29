import pickle as pkl

def pklload(x):
    with open(x, "rb") as f:
        return pkl.load(f)
