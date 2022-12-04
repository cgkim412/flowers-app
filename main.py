import pandas as pd
from color_table import prepare_table
from color_extraction import extract_colors
from PIL import Image
from flower_db import FlowerDatabase
import gradio as gr


db = FlowerDatabase()
table = prepare_table("flower_colors.csv")
symbols = pd.read_csv("all_symbols.csv").word.values.tolist()


def recommend_flowers(img, symbol):
    ref_colors = extract_colors(img)["rgb"]
    colors = table.batch_query(ref_colors)[:4]
    all_flowers = []
    for name, rgb in colors:
        flowers = db.get_flowers_by(color=name, symbol=symbol)
        all_flowers.append(flowers)
    return all_flowers


gr.Interface(
    fn=recommend_flowers,
    inputs=[gr.Image(type="pil"), gr.Dropdown(symbols)],
    outputs="json",
).launch(debug=True)

