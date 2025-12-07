import cv2
import numpy as np

def apply_gradient_coloring(edge_img, shade_img, gradient_colors=None, bg_color=(255,255,255), edge_strength=2):
    h, w = edge_img.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)

    kernel = np.ones((edge_strength, edge_strength), np.uint8)
    edge_img = cv2.dilate(edge_img, kernel, iterations=1)

    if gradient_colors is None:
        gradient_colors = [
            (0.0, (251,186,48)),
            (0.4, (252,114,53)),
            (0.6, (252,53,78)),
            (0.7, (207,54,223)),
            (0.8, (55,181,217)),
            (1.0, (62,182,218))
        ]

    gradient_map = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            t = (x + y) / (w + h)
            for i in range(len(gradient_colors)-1):
                if gradient_colors[i][0] <= t <= gradient_colors[i+1][0]:
                    t0, c0 = gradient_colors[i]
                    t1, c1 = gradient_colors[i+1]
                    a = (t - t0) / (t1 - t0)
                    color = (c0[0]*(1-a) + c1[0]*a,
                             c0[1]*(1-a) + c1[1]*a,
                             c0[2]*(1-a) + c1[2]*a)
                    gradient_map[y,x] = color
                    break

    alpha = np.maximum(edge_img, shade_img).astype(np.float32)/255.0
    output = (alpha[...,None] * gradient_map + (1-alpha[...,None])*bg_color).astype(np.uint8)

    return output
