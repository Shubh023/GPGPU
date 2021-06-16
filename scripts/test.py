import numpy as np

def main():
    pred_sk = np.loadtxt("resources/pred_sk.txt")
    pred_cpp = np.loadtxt("resources/pred_cpp.txt")
    print(f"Difference between sklearn and our implementation : "
          f"{np.sum(pred_sk != pred_sk) / len(pred_sk) * 100:.2f} %")

main()