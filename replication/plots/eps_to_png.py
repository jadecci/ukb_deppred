from pathlib import Path
import argparse

from PIL import Image


parser = argparse.ArgumentParser(
    description="Convert a .eps image file to a high resolution .png image file",
    formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, width=100))
parser.add_argument("in_file", type=Path, help="Absolute path to input file")
parser.add_argument("out_file", type=Path, help="Absolute path to output file")
args = parser.parse_args()

target_bounds = (1024, 1024)

# load at 10 times the default size
img = Image.open(args.in_file)
img.load(scale=10)
if img.mode in ("P", "1"):
    img = img.convert("RGB")

# resize image
ratio = min(target_bounds[0] / img.size[0], target_bounds[1] / img.size[1])
new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
img = img.resize(new_size, resample=Image.Resampling.LANCZOS)

img.save(args.out_file)
