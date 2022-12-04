import pandas as pd
import ujson as json
import os
from functools import cache
from glob import glob
import numpy as np


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def _build_db(json_dir, image_dir):
    columns = "name alt_names img_urls symbols seasons colors".split()
    jsons = glob(os.path.join(json_dir, "*.json"))
    jsons = [_read_json(js) for js in jsons]
    for js in jsons:
        _replace_image_url(js, image_dir)
    db = pd.DataFrame(jsons, columns=columns)
    return db


def _replace_image_url(json_dict, image_dir):
    flower_name = json_dict["name"].lower()
    num_imgs = len(json_dict["img_urls"])
    img_urls = [
        os.path.join(image_dir, flower_name, f"{i}.jpg").replace("\\", "/")
        for i in range(1, num_imgs + 1)
    ]
    json_dict["img_urls"] = img_urls


class FlowerDatabase:
    def __init__(self, json_dir="json", image_dir="images") -> None:
        self.db = _build_db(json_dir, image_dir)

    def _handle_query(self, field: str, value: str, bools_only: bool = False):
        if field not in ("colors", "symbols", "seasons"):
            raise ValueError
        if field == "colors":
            value = value.lower()
        else:
            value = value.capitalize()

        bools = self._boolean_query(field, value)
        if bools_only:
            return bools

        return self.db[bools].to_dict(orient="records")

    @cache
    def _boolean_query(self, field: str, value: str):
        return self.db.__getattr__(field).apply(lambda x: value in x).values

    def get_flowers_by_color(self, color: str):
        return self._handle_query("colors", color)

    def get_flowers_by_symbol(self, symbol: str):
        return self._handle_query("colors", symbol)

    def get_flowers_by_season(self, season: str):
        return self._handle_query("seasons", season)

    def get_flowers_by(self, color: str = None, season: str = None, symbol: str = None):
        n = len(self.db)
        all_true = np.ones(n).astype(bool)

        if color is None:
            c1 = all_true
        else:
            c1 = self._handle_query("colors", color, bools_only=True)

        if season is None:
            c2 = all_true
        else:
            c2 = self._handle_query("seasons", season, bools_only=True)

        if symbol is None:
            c3 = all_true
        else:
            c3 = self._handle_query("symbols", symbol, bools_only=True)

        return self.db[c1 & c2 & c3].to_dict(orient="records")


if __name__ == "__main__":

    db = FlowerDatabase()
    db.get_flowers_by_season("summer")
    db.get_flowers_by("red", "spring", "beauty")
