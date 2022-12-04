import pandas as pd
from color_table import prepare_table
from color_extraction import extract_colors
from PIL import Image
from flower_db import FlowerDatabase
import gradio as gr


db = FlowerDatabase()
table = prepare_table("flower_colors.csv")
seasons = "spring summer autumn winter".split()
symbols = [""] + pd.read_csv("all_symbols.csv").word.values.tolist()


def recommend_flowers(img, season, symbol):
    if not season:
        season = None
    if not symbol:
        symbol = None
    ref_colors = extract_colors(img)["rgb"]
    ref_colors = table.batch_query(ref_colors)[:4]
    all_flowers = []
    colors = []
    for name, rgb in ref_colors:
        flowers = db.get_flowers_by(color=name, season=season, symbol=symbol)[:3]
        all_flowers += flowers
        colors.append(name)
    return dict(extracted_colors=colors, recommended_flowers=all_flowers)


gr.Interface(
    fn=recommend_flowers,
    inputs=[gr.Image(type="pil"), gr.Dropdown(seasons), gr.Dropdown(symbols)],
    outputs="json",
).launch(debug=True)
