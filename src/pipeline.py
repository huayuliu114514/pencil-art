import cv2
import numpy as np
import bm3d
from shade import generate_shade_map, down_up_sample, merge_shade_texture
from edges import enhance_edges, convoluteY
from gradient import apply_gradient_coloring

def pencil_louvre_filter(image_path, texture_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    texture = cv2.imread(texture_path,0)
    texture = cv2.resize(texture, (gray.shape[1], gray.shape[0]))

    shade = generate_shade_map(gray)
    shade = down_up_sample(shade)
    shade_final = merge_shade_texture(shade, texture)

    gray = enhance_edges(gray, "sharpen1")

    sobel_x = [-1,0,1,-2,0,2,-1,0,1]
    sobel_y = [1,2,1,0,0,0,-1,-2,-1]

    edge_x = convoluteY(gray, sobel_x)
    edge_y = convoluteY(gray, sobel_y)
    edge = np.sqrt(edge_x**2 + edge_y**2).astype(np.uint8)

    # BM3D
    e = edge.astype(np.float32)/255.0
    e = bm3d.bm3d(e, 0.1)*255
    e = e.astype(np.uint8)

    final_img = apply_gradient_coloring(e, shade_final)

    return final_img
