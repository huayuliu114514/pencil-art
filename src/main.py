from pipeline import pencil_louvre_filter
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="input image")
parser.add_argument("--output", default="output.png")
parser.add_argument("--texture", default="textures/pencil-texture.jpg")
args = parser.parse_args()

result = pencil_louvre_filter(args.input, args.texture)
cv2.imwrite(args.output, result)

print("Saved:", args.output)
