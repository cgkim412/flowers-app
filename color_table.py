import numpy as np
import pandas as pd
from PIL import Image, ImageColor
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab, rgb2xyz, xyz2lab
from skimage.io import imread, imshow

from color_extraction import extract_colors


class ColorTable:
    def __init__(self, names, values):
        assert len(names) == len(values)
        self.names = np.array(names)
        self.values = np.array(values, dtype=np.uint8)
        self.lab_values = rgb2lab(self.values)

    def query(self, rgb_value):
        rgb = np.array(rgb_value, dtype=np.uint8)
        if rgb.ndim != 1 or rgb.shape != (3,):
            raise ValueError("Single RGB value is expected.")

        lab = rgb2lab(rgb).reshape(1, 3)
        distances = deltaE_ciede2000(lab, self.lab_values)
        best_match_idx = distances.argmin()

        name = self.names[best_match_idx]
        value = self.values[best_match_idx]
        return name.item(), tuple(value.tolist())

    def batch_query(self, rgb_values):
        mapped_colors = [self.query(rgb) for rgb in rgb_values]
        return mapped_colors


def prepare_table(csv_path):
    df = pd.read_csv(csv_path)
    df["rgb"] = df["code"].apply(lambda x: ImageColor.getcolor(x, "RGB"))
    table = ColorTable(df["name"].tolist(), df["rgb"].tolist())
    return table


if __name__ == "__main__":

    img = Image.open("ref2.jpg")
    centroids = extract_colors(img)
    ref_colors = centroids["rgb"]

    table = prepare_table("flower_colors.csv")
    result = table.batch_query(ref_colors)

    print(f"Extracted colors:\n {ref_colors}")
    print(f"Query result: \n {result}")

    _ = imshow(centroids["rgb"][None, ...])
