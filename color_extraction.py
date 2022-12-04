import numpy as np
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from skimage.io import imshow
from sklearn.cluster import KMeans


def float2int(x):
    return (255 * x).round().astype(np.uint8)


def extract_colors(img: Image.Image, n_colors=8, downsample_size=128):
    assert img.mode == ("RGB")

    rgb = img.convert("RGB").resize((downsample_size, downsample_size))
    rgb = np.array(img)
    lab = rgb2lab(rgb)
    X = lab.reshape(-1, 3)

    kms = KMeans(n_clusters=n_colors, n_init=3, random_state=9)
    kms.fit(X)

    _, counts = np.unique(kms.labels_, return_counts=True)
    order = counts.argsort()[::-1]

    lab_centers = kms.cluster_centers_[order].copy()
    rgb_centers = float2int(lab2rgb(lab_centers))
    return dict(rgb=rgb_centers, lab=lab_centers)


if __name__ == "__main__":

    img = Image.open("ref2.jpg")
    centroids = extract_colors(img)

    img.resize([256, 256])
    _ = imshow(centroids["rgb"][None, ...])
