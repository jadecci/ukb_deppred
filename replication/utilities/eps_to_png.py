from pathlib import Path
import argparse

from PIL import Image

parser = argparse.ArgumentParser(
    description="Convert .eps image to high resolution .png image",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("in_file", type=Path, help="Absolute path to input image file")
parser.add_argument("out_file", type=Path, help="Absolute path to output image file")
args = parser.parse_args()

img = Image.open(args.in_file)
img.load(scale=10)
if img.mode in ("P", "1"):
    img = img.convert("RGB")

ratio = min(1024 / img.size[0], 1024 / img.size[1])
new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
img = img.resize(new_size)

img.save(args.out_file)
