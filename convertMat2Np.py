import numpy as np
from scipy.io import loadmat


def convertmat(filemat):
    return loadmat(filemat)


if __name__ == "__main__":
    test_path = r"C:\Users\renau\Dropbox (MIT)\1_PhD\Code\Machine Learning\6s198_project\sketch2shape\data\models\chair.mat"
    print(np.min(convertmat(test_path)['inputs']))
    print(np.max(convertmat(test_path)['inputs']))
