import numpy as np
from scipy.spatial.distance import cdist



def test_l2_sq():
    centroids = np.loadtxt("resources/centroids.txt")
    desc = np.loadtxt("resources/desc.txt")
    #res = desc - centroids.T
    res = cdist(desc, centroids, 'sqeuclidean')
    print(res.shape)    
    np.savetxt("resources/res.txt", np.argmin(res, axis=1), fmt="%d")
    #np.savetxt("resources/test.txt", res, fmt="%.5f")

def main():
    pred_sk = np.loadtxt("resources/pred_sk.txt")
    pred_cpp = np.loadtxt("resources/pred_cpp.txt")
    print(f"Difference between sklearn and our implementation : "
          f"{np.sum(pred_sk != pred_cpp) / len(pred_sk) * 100:.2f} %")

main()
#test_l2_sq()