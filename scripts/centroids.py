from multiprocessing import cpu_count, Pool
from skimage.util import view_as_blocks
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import cv2
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt


def resize_img(img):
    h, w = img.shape[:2]
    h2 = 16 * round(h / 16) 
    w2 = 16 * round(w / 16) 
    if h != h2 or w != w2:
        img = cv2.resize(img, (w2, h2), interpolation=cv2.INTER_CUBIC)
    return img

def extract_descriptor(patch):
    patch = np.pad(patch, ((1,1),(1,1)), 'constant')
    textons = np.zeros((16, 16), dtype=np.ubyte)
    for i in range(1, patch.shape[0] - 1):
        for j in range(1, patch.shape[1] - 1):
            cell = patch[i-1:i+2, j-1:j+2]
            center = cell[1, 1]
            borders = np.concatenate([
                cell[0,:],
                np.array([cell[1,0], cell[1,-1]]),
                cell[-1,:]
            ])
            texton = (borders > center).astype(int)
            texton = int(''.join(list(map(str, texton))), 2)
            textons[i-1, j-1] = texton
            break
    return np.histogram(textons, bins=256)[0]

def fit_kmeans(descriptors, k):
    #kmeans = MiniBatchKMeans(n_clusters=k, random_state=0)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(descriptors)
    centroids = kmeans.cluster_centers_

    print("Saving output files.")
    with open(f"model_{k}.pickle", "wb") as f:
        pickle.dump(kmeans, f)

    np.savetxt(f"centroids_{k}.txt", centroids, fmt="%.18f")
    np.savetxt(f"centroids_t_{k}.txt", centroids.T, fmt="%.18f")

    return kmeans


def main():
    files = glob.glob("pics/*.jpg")
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in files]

    patches = []
    for img in images:
        img = resize_img(img)
        im_p = view_as_blocks(img, (16, 16))
        im_p = im_p.reshape(im_p.shape[0] * im_p.shape[1], 16, 16)
        patches.append(im_p)

    patches = np.vstack(patches)
    print(f"Patches shape : {patches.shape}")
    tot = patches.shape[0]
    patches = patches[np.random.choice(tot, int(tot * 0.2), replace=False)]

    print(f"Computing descriptors for {patches.shape[0]} patches.")

    # Sequential version
    #descriptors = []
    #for patch in tqdm(patches[:1]):
    #    descriptors.append(extract_descriptor(patch))
    #descriptors = np.array(descriptors)

    # Parallel version
    pool = Pool(cpu_count() - 1)
    descriptors = pool.map(extract_descriptor, patches)

    np.savetxt(f"desc.txt", descriptors, fmt="%u")

    print("Fitting KMeans.")
    fit_kmeans(descriptors, k=16)
    fit_kmeans(descriptors, k=32)
    fit_kmeans(descriptors, k=64)
    fit_kmeans(descriptors, k=128)

main()