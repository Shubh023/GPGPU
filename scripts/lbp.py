from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import cv2

patch_size = 16


def extract_patches(img):
    h, w = img.shape[:2]
    w = round(w, patch_size)
    h = round(h, patch_size)
    dim = (w, h)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
    print(resized.shape)

    img_patches = []
    for r in range(0, resized.shape[0] - patch_size, patch_size):
        for c in range(0, resized.shape[1] - patch_size, patch_size):
            patch = resized[r:r+patch_size, c:c+patch_size]
            img_patches.append(patch)
    img_patches = np.array(img_patches)

    return img_patches


def extract_textons(patch):

    def extract_elements(a):
        n = a.shape[0]
        r = np.minimum(np.arange(n)[::-1], np.arange(n))
        return a[np.minimum(r[:,None],r) < 1]

    test_patch_pad = np.pad(patch, ((1,1),(1,1)), 'constant')
    textons = []

    for i in range(patch_size):
        for j in range(patch_size):
            cell = test_patch_pad[i:i+3,j:j+3]
            cell_center = cell[1][1]
            #print(cell)
            compared_pixels = (extract_elements(cell) > cell_center).astype(int)
            textons.append(compared_pixels)

    return np.array(textons)


def hist_from_texton(textons):
    hist = np.zeros(256)
    for i in textons:
        val = int(''.join(list(map(str, i))), 2)  # binary to Int
        hist[val] += 1
    return hist


def assign_clusters(descriptors):
    kmeans = KMeans(n_clusters=32, random_state=0)
    pred = kmeans.fit_predict(descriptors)
    return pred, kmeans.cluster_centers_


from skimage.feature import local_binary_pattern

def main():
    img = cv2.imread("resources/beans.jpg", cv2.IMREAD_GRAYSCALE)
    patches = extract_patches(img)

    descriptors = np.array([hist_from_texton(t) for t in p_textons])
    pred, centroids = assign_clusters(descriptors)
    np.savetxt("resources/pred_sk.txt", pred, fmt="%u")
    np.savetxt("resources/centroids.txt", centroids, fmt="%.18f")
    np.savetxt("resources/centroids_t.txt", centroids.T, fmt="%.18f")
    np.savetxt("resources/desc.txt", descriptors, fmt="%u")


main()